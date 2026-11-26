import base64
from mimetypes import guess_type

from langchain_community.tools.steamship_image_generation.tool import SUPPORTED_IMAGE_SIZES
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from minio.deleteobjects import DeleteObject

from app.infra.object_storage.minio_gateway import minio_gateway

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import IMAGE_CONTEXT_CHAR_LENGTH,SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from minio import Minio
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

def summary_image_info(image_info:list[tuple[str,Path,tuple[str,str]]],root_folder:str) -> dict[str,str]:
    # 3.1 数据准备工作 初始化一个视觉模型对象  | 定义一个接受数据的字典
    images_summary:dict[str,str] = {}
    vision_model = llm_provider.vision_chat(model="gpt-4o")
    # 3.2 环处理每一张图片  for str,Path,context in list[tuple[str,Path,tuple[str,str]]]
    for image_name,image_path_obj,image_context in image_info:
        # 3.3 封装模型提示词信息
        # 3.3.1 图片的描述 包裹上下文信息一段提示词
        image_des :str = load_prompt("image_summary",image_context=image_context,root_folder=root_folder)
        image_type:str = guess_type(image_name)[0]
        image_base64:str = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type":"text","text":image_des},
                {
                    "type":"image_url",
                    "image_url":{"url":f"data:{image_type};base64,{image_base64}"},
                },
            ]
        )
        # 3.4 封装调用链：直接model.invoke|调用链 提示词 |模型|解析器 OutParser()
        chains = vision_model | StrOutputParser()
        # 3.5 使用调用链进行调用和获取结果
        summary:str = chains.invoke([message])
        images_summary[image_name] = summary
        logger.debug(f"图片:{image_name},已经通过视觉模型进行意图识别，结果为{summary}")
    return images_summary

def upload_image_to_minio(image_info:list[tuple[str,Path,tuple[str,str]]],stem)  ->dict[str,str]:
    """
        4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
        4.2 根据桶信息查询对应旧图片对象
        4.3 批量删除旧图片对象
        4.4 循环处理每一张图片  for image_name,image_path_obj,_ in image_info:
            4.5 单次循环的异常处理,失败,跳过图片即可,进行下一张!
            4.6 调用minio_client.fput方法将图片上传到minio服务器
            4.7 根据信息调用infra构建图片的访问地址
            4.8 将图片访问地址存储到dict image_name | image_url
        4.9 最终返回dict数据
    """
    # 4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
    image_url:dict[str,str] = {}
    minio_client:Minio = minio_gateway.client
    # 4.2 根据桶信息查询对应旧图片对象
    objects:list  =  minio_client.list_objects(
         bucket_name=minio_gateway.bucket_name,
         # 桶名  -> /图片前缀/文件名/ <- 图片.png
         # 注意：要求必须不能使用/开始
         prefix=minio_gateway.image_dir[1:] + "1" + stem ,
         recursive=True
    )
     # 4.3 批量删除旧图片对象
    if objects:
        # 查询到了之前文件有对应图片，进行删除
        delete_object_list = [DeleteObject(obj.object.name) for obj in objects ]
        errors = minio_client.remove_objects(
            bucket_name=minio_gateway.bucket_name,
            delete_object_list=delete_object_list
        )
        for error in errors:
            logger.warning(f"删除{stem}文件夹下的图片出现问题，问题：{error}")
    # 4.4 循环处理每一张图片  for image_name,image_path_obj,_ in image_info:
    for image_name,image_path_obj,_ in image_info:
        # 4.5 单次循环的异常处理,失败,跳过图片即可,进行下一张!
        try:
            # 4.6 调用minio_client.fput方法将图片上传到minio服务器
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=minio_gateway.image_dir + "/" +stem + "/" + image_name,                   #object_name -> 决定我们在桶中的文件结构！
                file_path=str(image_path_obj),     #Path ->str
                content_type=guess_type(image_name)[0]
            )
            # 不报错，成功
            # 4.7 根据信息调用infra构建图片的访问地址
            url:str = minio_gateway.build_image_url(stem,image_name)
            # 4.8 将图片访问地址储存到dict image_name | image_url
            image_url[image_name] = url
            logger.debug(f"{image_name}已经传递到了minio文件服务器，访问地址为：{url}")
        except Exception as e:
            logger.exception(f"{image_name}图书上传失败，跳过本次，进行下一张图片处理！")
            continue

    # 4.9 最终返回dict数据
    logger.debug(f"已经{stem}所有图片的上传，上传的数量为：{len(image_url)}")
    return image_url


"""
5. 完成md_content内部图片标签的替换,获得新的md_content内容
        目标: md_content内容进行替换 返回替换后的md_content ![](./) -> ![网络地址](http...)
        replace_old_md_content(md_content:str,images_url:dict[str,str],images_summary:dict[str,str]) -> str:
        5.1 循环images_url可以获取每张图片的image_name / 网络地址
        5.2 可以根据image_name去images_summary获取对应的summary描述
        5.3 定义提前编译查询原有图片标签的正则 
        5.4 使用正则的替换函数即可    替换后的内容 md_content  =  编译的对象.sub("替换入的内容" f"![summary](url)" ,"什么中查找替换" -> md_content)
        5.5 返回替换后的md_content
6. 将新md_content写到本地磁盘,持久化
        目标: 文件名/文件名.md [old] 新的md_content->写到本地磁盘!!  文件名_new.md,最终返回一个新的md_path:str
        backup_md_return_path(md_content:str,md_path_obj:Path)
        6.1 获取新备份位置的Path 文件名_new.md  md_path_obj.with_name | rename
        6.2 向新的path对象写入新的md_content md_path_obj.write_text
        6.3 新的path对象转成字符串返回即可
7. 更新state信息md_path -> 新的地址  md_content -> 新的内容
        state['md_path'] = 新地址
        state['md_content'] = 新content内容
8. 返回state
        return state
"""
def replace_old_md_content(md_content:str,image_url:dict[str,str],images_summary:dict[str,str])  ->str:
    # 5.1 循环image_url可以获取每张图片的image_name / 网络地址
    for image_name,url in image_url.items():
        # 5.2可以根据image_name去images_summary获取对应的summary描述
        summary = images_summary.get(image_name)
        # 5.3定义提前编译查询原有图片标签的正则 re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)"
        image_re = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # 5.4 使用正则的替换函数即可 替换后的内容 md_content = 编译的对象.sub("替换入的内容" f"![summary](url)", "什么中查找替换" -> md_content)
        md_content:str = image_re.sub(lambda _:f"![{summary}]({url})",md_content)
        # 5.5 返回替换后的md_content
    return md_content

def backup_md_return_path(md_content:str,md_path_obj:Path)   ->str:
# 6. 将新md_content写到本地磁盘,持久化
#    目标: 文件名/文件名.md [old] 新的md_content->写到本地磁盘!!  文件名_new.md,最终返回一个新的md_path:str
#    backup_md_return_path(md_content:str,md_path_obj:Path)
#    6.1 获取新备份位置的Path 文件名_new.md  md_path_obj.with_name | rename
     new_md_path_obj:Path = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
#    6.2 向新的path对象写入新的md_content md_path_obj.write_text
     new_md_path_obj.write_text(data=md_content,encoding="utf-8")
#    6.3 新的path对象转成字符串返回即可
     return str(new_md_path_obj)



def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    # 1.获取参数并且校验
    md_content, md_path_obj, images_path_obj = validates_and_get_data(state)

    if not list(images_path_obj.iterdir()):
        logger.info(f"{md_path_obj}中没有引用任何图片，无需后续图片处理，直接跳转到下一个节点！！")
        return state
    # 2.调用方法进行图片信息识别
    images_info_list: list[tuple[str, Path, tuple[str, str]]] = scan_iamges_info(images_path_obj, md_content)
    # 3.调用模型进行图片内容总结
    # 3.调用模型进行图片内容总结
    if images_info_list:
        images_summary: dict[str, str] = summary_image_info(images_info_list, md_path_obj.stem)
    else:
        logger.info("没有需要总结的图片，跳过。")
        images_summary = {}
    # 4.调用图片上传，获取图片上传地址
    images_url :dict[str,str] = upload_image_to_minio(images_info_list,md_path_obj.stem)
    # 5.进行md_content内容替换
    new_md_content:str = replace_old_md_content(md_content,images_url,images_summary)
    # 6.进行new_md_content数据备份
    new_md_content:str = backup_md_return_path(md_content,md_path_obj)
    # 7. 更新state
    state['md_content'] = new_md_content
    # 8.返回结果
    return state
