from langchain_community.tools.steamship_image_generation.tool import SUPPORTED_IMAGE_SIZES

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import IMAGE_CONTEXT_CHAR_LENGTH
from app.shared.runtime.logger import logger
import re

"""
1.获取核心参数和数据校验
目标：state   获取：md_path_obj  images_path_obj  md_content
validates_and_get_data(state:ImportGraphState)  -> tuple[str,Path,Path]
1.1 获取md_path参数从state
1.2 md_path的非空校验
1.3 转成md_path_obj Path对象，做存在性校验！is_file()
1.4 使用md_path对象读取md_content,更新state
1.5 md_path_obj 可以获取images path地址
1.6 顺序返回 return  md_content
"""


def validates_and_get_data(state: ImportGraphState) -> tuple[str, Path, Path]:
    # 1.1 获取md_path参数从state
    md_path: str = state.get("md_path")
    # 1.2 md_path的非空校验
    if not md_path:
        logger.error(f"md_path内容为空，业务无法继续，提前终止！")
        raise ValueError(f"md_path内容为空，业务无法继续，提前终止！")
    # 1.3 转成md_path_obj Path对象，做存在性校验！is_file()
    md_path_obj: Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path地址有值:{md_path},但是没有对应的文件，业务无法继续，提前终止！")
        raise FileNotFoundError(f"md_path地址有值:但是没有对应的文件，业务无法继续，提前终止！")
    # 1.4 使用md_path对象读取md_content,更新state
    md_content: str = md_path_obj.read_text(encoding="utf-8")
    state["md_content"] = md_content
    # 1.5 md_path_obj 可以获取images path地址
    images_path_obj: Path = md_path_obj.parent / "images"
    # 1.6 顺序返回 return  md_content
    return md_content, md_path_obj, images_path_obj


def scan_iamges_info(images_path_obj: Path, md_content: str) -> list[tuple[str, Path, tuple[str, str]]]:
    # 2.1 获取images下的所有图片Path对象 [列表/生成器]
    generator = images_path_obj.iterdir()
    images_info_list: list[tuple[str, Path, tuple[str, str]]] = []
    # 2.2 循环每一张图片对象
    for image_file_obj in generator:
        if image_file_obj.suffix not in SUPPORTED_IMAGE_SIZES:
            logger.debug(f"{image_file_obj.name}不是图片，跳过，进行下一张处理!")
            continue
        # 2.2.1 定义这个图片的正则规则
        image_name: str = image_file_obj.name
        match = re.search(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)", md_content)
        # 2.2.3 检查是否匹配 Match是否为空
        if not match:
            logger.debug(f"{image_name}没有在md_content被使用，跳过，进行下一张处理！")
            continue
        # 2.2.4 获取对应上下文信息
        start = match.start()
        end = match.end()

        pre_context: str = md_content[max(start - IMAGE_CONTEXT_CHAR_LENGTH, 0):start]
        post_context: str = md_content[end + 1:min(end + IMAGE_CONTEXT_CHAR_LENGTH, len(md_content))]
        images_info_list.append(
            (
                image_name,
                image_file_obj,
                (
                    pre_context,
                    post_context
                )
            )

        )
    logger.info(f"已经完成了{len(images_info_list)}张图片内容的匹配！")
    return images_info_list


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    # 1.获取参数并且校验
    md_content, md_path_obj, images_path_obj = validates_and_get_data(state)
    # 调用方法进行图片信息识别
    images_info_list: list[tuple[str, Path, tuple[str, str]]] = scan_iamges_info(images_path_obj, md_content)
    return state
