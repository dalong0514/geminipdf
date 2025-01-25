import os, time
import re
from typing import List, Tuple, Optional, Dict
import logging
import google.generativeai as genai
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures


class GeminiAgent:
    def __init__(self, model_name: str, **kwargs):
        logging.info(f"Initializing GeminiAgent with model: {model_name}")
        self.model = genai.GenerativeModel(model_name)

    def run(self, messages, display: bool = False):
        if isinstance(messages, str):
            msgs_for_gemini = [messages]
        elif isinstance(messages, list):
            msgs_for_gemini = []
            for m in messages:
                if isinstance(m, dict) and 'image' in m:
                    image_path = m['image']
                    with open(image_path, 'rb') as f:
                        raw_data = f.read()
                    b64_data = base64.b64encode(raw_data).decode('utf-8')
                    msgs_for_gemini.append({
                        "mime_type": "image/png",
                        "data": b64_data
                    })
                else:
                    msgs_for_gemini.append(str(m))
        else:
            msgs_for_gemini = [str(messages)]

        if display:
            logging.info(f"Sending messages to Gemini: {msgs_for_gemini}")

        response = self.model.generate_content(msgs_for_gemini)
        if display:
            logging.info(f"Received response from Gemini: {response.text}")

        return response.text

def merge_lines(text: str) -> str:
    """
    Use simple regex to replace single line breaks with spaces, while preserving consecutive empty lines as paragraphs.
    You can make it more complex as needed, such as handling hyphens, etc.
    """
    # First convert \r\n to \n
    text = text.replace('\r\n', '\n')
    
    # Treat 2 or more consecutive newlines as paragraph separators, using a special marker <PARA> as placeholder
    text = re.sub(r'\n\s*\n+', '<PARA>', text)

    # Replace single newlines with spaces within the same paragraph
    text = re.sub(r'\n+', ' ', text)

    # Restore <PARA> to actual paragraph separators (using two newlines here)
    text = text.replace('<PARA>', '\n\n')

    return text

# This Default Prompt Using Chinese and could be changed to other languages.
DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
4. 段落与段落之间用空行隔开。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
DEFAULT_RECT_PROMPT = """图片中用红色框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""
DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""


def _is_near(rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 20) -> bool:
    """
    Check if two rectangles are near each other if the distance between them is less than the target.
    """
    return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance


def _is_horizontal_near(rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 100) -> bool:
    """
    Check if two rectangles are near horizontally if one of them is a horizontal line.
    """
    result = False
    if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
        if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
            result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
    return result


def _union_rects(rect1: BaseGeometry, rect2: BaseGeometry) -> BaseGeometry:
    """
    Union two rectangles.
    """
    return sg.box(*(rect1.union(rect2).bounds))


def _merge_rects(rect_list: List[BaseGeometry], distance: float = 20, horizontal_distance: Optional[float] = None) -> \
        List[BaseGeometry]:
    """
    Merge rectangles in the list if the distance between them is less than the target.
    """
    merged = True
    while merged:
        merged = False
        new_rect_list = []
        while rect_list:
            rect = rect_list.pop(0)
            for other_rect in rect_list:
                if _is_near(rect, other_rect, distance) or (
                        horizontal_distance and _is_horizontal_near(rect, other_rect, horizontal_distance)):
                    rect = _union_rects(rect, other_rect)
                    rect_list.remove(other_rect)
                    merged = True
            new_rect_list.append(rect)
        rect_list = new_rect_list
    return rect_list


def _adsorb_rects_to_rects(source_rects: List[BaseGeometry], target_rects: List[BaseGeometry], distance: float = 10) -> \
        Tuple[List[BaseGeometry], List[BaseGeometry]]:
    """
    Adsorb a set of rectangles to another set of rectangles.
    """
    new_source_rects = []
    for text_area_rect in source_rects:
        adsorbed = False
        for index, rect in enumerate(target_rects):
            if _is_near(text_area_rect, rect, distance):
                rect = _union_rects(text_area_rect, rect)
                target_rects[index] = rect
                adsorbed = True
                break
        if not adsorbed:
            new_source_rects.append(text_area_rect)
    return new_source_rects, target_rects


def _parse_rects(page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    """
    Parse drawings in the page and merge adjacent rectangles.
    """

    # Extract drawing content
    drawings = page.get_drawings()

    # Ignore horizontal lines shorter than 30 pixels
    is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 1 and abs(x['rect'][2] - x['rect'][0]) < 30
    drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

    # Convert to shapely rectangles
    rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

    # Extract image regions
    images = page.get_image_info()
    image_rects = [sg.box(*image['bbox']) for image in images]

    # Merge drawings and images
    rect_list += image_rects

    merged_rects = _merge_rects(rect_list, distance=10, horizontal_distance=100)
    merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

    # Separate large and small text areas for processing: merge large text areas with small ones, merge small text areas when close
    is_large_content = lambda x: (len(x[4]) / max(1, len(x[4].split('\n')))) > 5
    small_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if not is_large_content(x)]
    large_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if is_large_content(x)]
    _, merged_rects = _adsorb_rects_to_rects(large_text_area_rects, merged_rects, distance=0.1) # Fully intersecting
    _, merged_rects = _adsorb_rects_to_rects(small_text_area_rects, merged_rects, distance=5) # Merge when close

    # Merge rectangles with themselves again
    merged_rects = _merge_rects(merged_rects, distance=10)

    # Filter out small rectangles
    merged_rects = [rect for rect in merged_rects if rect.bounds[2] - rect.bounds[0] > 20 and rect.bounds[3] - rect.bounds[1] > 20]

    return [rect.bounds for rect in merged_rects]


def _parse_pdf_to_images(pdf_path: str, output_dir: str = './') -> List[Tuple[str, List[str]]]:
    """
    Parse PDF to images and save to output_dir.
    """
    # Open PDF file
    pdf_document = fitz.open(pdf_path)
    image_infos = []

    for page_index, page in enumerate(pdf_document):
        logging.info(f'parse page: {page_index}')
        rect_images = []
        rects = _parse_rects(page)
        for index, rect in enumerate(rects):
            # Ensure rectangle coordinates are within page bounds
            x0 = max(0, rect[0])
            y0 = max(0, rect[1])
            x1 = min(page.rect.width, rect[2])
            y1 = min(page.rect.height, rect[3])
            
            # Skip if rectangle is invalid
            if x0 >= x1 or y0 >= y1:
                continue
                
            fitz_rect = fitz.Rect(x0, y0, x1, y1)
            # Save page as image
            try:
                pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
                name = f'{page_index}_{index}.png'
                pix.save(os.path.join(output_dir, name))
                rect_images.append(name)
                # Draw red rectangle on the page
                big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
                # Hollow rectangle
                page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)
                # Write the rectangle index name at the top-left corner of the rectangle with some offset
                text_x = fitz_rect.x0 + 2
                text_y = fitz_rect.y0 + 10
                text_rect = fitz.Rect(text_x, text_y - 9, text_x + 80, text_y + 2)
                # Draw white background rectangle
                page.draw_rect(text_rect, color=(1, 1, 1), fill=(1, 1, 1))
                # Insert text with white background
                page.insert_text((text_x, text_y), name, fontsize=10, color=(1, 0, 0))
            except Exception as e:
                logging.warning(f"Failed to save image {name}: {str(e)}")
                continue
        page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_image = os.path.join(output_dir, f'{page_index}.png')
        page_image_with_rects.save(page_image)
        image_infos.append((page_image, rect_images))

    pdf_document.close()
    return image_infos

def _gemini_parse_images(
        image_infos: List[Tuple[str, List[str]]],
        prompt_dict: Optional[Dict] = None,
        output_dir: str = './',
        api_key: Optional[str] = None,
        model: str = 'gemini-1.5-flash-002',
        verbose: bool = False,
        gemini_worker: int = 1,
        request_delay: float = 2.0,  # Request interval time, default 2 seconds
        **args
) -> str:
    """
    Parse given image information using Google Gemini model and output Markdown content.
    """
    # Handle custom prompts from prompt_dict
    if isinstance(prompt_dict, dict) and 'prompt' in prompt_dict:
        prompt = prompt_dict['prompt']
        logging.info("prompt is provided, using user prompt.")
    else:
        prompt = DEFAULT_PROMPT
        logging.info("prompt is not provided, using default prompt.")

    if isinstance(prompt_dict, dict) and 'rect_prompt' in prompt_dict:
        rect_prompt = prompt_dict['rect_prompt']
        logging.info("rect_prompt is provided, using user prompt.")
    else:
        rect_prompt = DEFAULT_RECT_PROMPT
        logging.info("rect_prompt is not provided, using default prompt.")

    if isinstance(prompt_dict, dict) and 'role_prompt' in prompt_dict:
        role_prompt = prompt_dict['role_prompt']
        logging.info("role_prompt is provided, using user prompt.")
    else:
        role_prompt = DEFAULT_ROLE_PROMPT
        logging.info("role_prompt is not provided, using default prompt.")

    # Initialize GeminiAgent
    genai.configure(api_key=api_key, transport="rest")
    gemini_agent = GeminiAgent(model_name=model, **args)

    def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
        logging.info(f'Gemini parse page: {index}')
        time.sleep(request_delay)  # Add delay before each request

        page_image, rect_images = image_info
        # Assemble content to be sent to the model
        local_prompt = role_prompt + "\n" + prompt
        if rect_images:
            local_prompt += rect_prompt + ", ".join(rect_images)

        messages = [local_prompt, {'image': page_image}]

        # Retry logic
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                content = gemini_agent.run(messages, display=verbose)
                content = merge_lines(content)
                print('OK...')

                # 如果有多余的 ```markdown 等，需要进行清理
                if '```markdown' in content:
                    content = content.replace('```markdown\n', '')
                    last_backticks_pos = content.rfind('```')
                    if last_backticks_pos != -1:
                        content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]

                return index, content
                
            except Exception as e:
                retry_count += 1
                if "429" in str(e) and retry_count < max_retries:
                    wait_time = 60 * retry_count  # 每次重试等待时间递增
                    logging.warning(f"Rate limit exceeded, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to process page {index} after {retry_count} attempts: {str(e)}")
                    return index, f"Error processing page {index}: {str(e)}"

    output_path = os.path.join(output_dir, 'output.md')
    # Process each page image in parallel
    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gemini_worker) as executor:
        futures = [executor.submit(_process_page, idx, info) for idx, info in enumerate(image_infos)]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            contents[index] = content
            # Write results to output.md
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(content)
    output_text = '\n\n'.join(contents)

    return output_text

def parse_pdf(
        pdf_path: str,
        output_dir: str = './',
        prompt: Optional[Dict] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gemini-2.0-flash-exp',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> Tuple[str, List[str]]:
    """Parse PDF file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle null/empty value cases
    if prompt is None:
        prompt = {}

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    
    content = _gemini_parse_images(
        image_infos=image_infos,
        output_dir=output_dir,
        prompt_dict=prompt,
        api_key=api_key,
        model=model,
        verbose=verbose,
        **args
    )

    all_rect_images = []
    # Clean up temporary image files
    for _, rect_images in image_infos:
        all_rect_images.extend(rect_images)
    
    return content, all_rect_images