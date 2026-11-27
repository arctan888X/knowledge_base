import json

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN
from app.shared.runtime.logger import logger
from typing import Any
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def validates_and_get_data(state:ImportGraphState) -> tuple[str,str,Path]:
    # 1.1 获取相关参数 md_content / file_title / md_path
    md_content:str = state.get("md_content")
    md_path   :str = state.get("md_path")
    file_title:str = state.get("file_title")
    # 1.2 非空校验
    if not md_path:
        logger.error(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
        raise ValueError(f"md_path执行的md文件为空,业务无法继续,提前终止!!")
    md_path_obj:Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")
        raise FileNotFoundError(f"md_path地址为:{md_path},但是没有对应的文件!业务无法继续,提前终止!!")
    if not md_content:
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
            raise ValueError(f"md_content为空,读取md_path依然为空,业务无法继续,提前终止!!")
        state['md_content'] = md_content
    if not file_title:
        file_title = md_path_obj.stem
        logger.warning(f"file_title为空,给与默认值:{file_title},业务继续!")
    # 1.3 返回对应的结果
    return md_content,file_title,md_path_obj


"""
 document_split_by_title(md_content:str) -> list[dict[str,Any]]:
        2.1 先创建存储数据容器 [ 1.存储历史标题的数据 list[dict[str,Any]]   2. current_title:str  3. current_title_lines:list[str] ]
        2.2 对清洗后的md_content进行按行切割 \n -> list[str]
        2.3 定义正则匹配对应的标题
        2.4 循环md_content切割的行 逐行处理[标题/普通行]
            2.5 查看是不是标题
            2.6 是-> 结算上一赛季 / 记录新的赛季
            2.7 不是 -> 普通行 -> 追加到 current_title_lines 继续下一行处理即可
        2.8 跳出循环,最后进行数据结算 [因为最后一个标题没有人触发结束]
        2.9 返回结果
"""
def document_split_by_title(md_content:str) -> list[dict[str,Any]]:
    # 2.1 先创建存储数据容器 [ 1.存储历史标题的数据 list[dict[str,Any]]   2. current_title:str  3. current_title_lines:list[str] ]
    chunks_title: list[dict[str, Any]] = []
    current_title: str|None            = None
    current_title_lines: list[str]     = []
    is_code:bool                       = False
    # 2.2 对清洗后的md_content进行按行切割 \n -> list[str]
    md_lines: list[str] = md_content.split("\n")
    # 2.3 定义正则匹配对应的标题
    """
    ^ 从头匹配
    $ 结尾匹配
    \s 空格
    *  量次 0-n
    +  量次 1-n
    {1,2} 固定量次  [] 1-2
    """
    title_re = re.compile(r"^\s*#{1,6}\s.+")
    # 2.4 循环md_content切割的行 逐行处理[标题/普通行]
    for line in md_lines:
        # 去除首尾空白字符（结合之前讨论的，这一步把空行变成了 ""）
        line = line.strip()
        # 跳过纯空行，防止往列表里塞无意义的空字符串
        if not line:
            logger.debug(f"当前行为空行，跳出进行进行下一行")
            continue
        logger.debug(f"当前行为有内容正常行，正常处理")

        if line.startswith("~~~") or line.startswith("```"):
            current_title_lines.append(line)
            is_code = not is_code
            continue

        # 2.5 查看是不是标题
        match = title_re.match(line)
        # 2.6 是-> 结算上一赛季 / 记录新的赛季
        if not is_code and match:
            # 结算：如果当前段落有内容，就存入总容器（过滤掉初始状态空内容的情况）
            if current_title and len(current_title_lines) > 1:
                chunks_title.append(
                    {
                     "title": current_title,
                    "content": "\n".join(current_title_lines)
                }
            )

            #孤儿数据
            if not current_title and len(current_title_lines) > 0:
                current_title_lines.append(line)
                current_title  = line
                continue
            # 更新本赛季
            current_title = line
            current_title_lines = [line]  # 标题是当前段落的开始

        else:
        # 2.7 不是 -> 普通行 -> 追加到 current_title_lines 继续下一行处理即可
            current_title_lines.append(line)
    # 2.8 跳出循环,最后进行数据结算 [因为最后一个标题没有人触发结束]
    if current_title and len(current_title_lines) > 1:
        chunks_title.append(
            {
            "title": current_title,
            "content": "\n".join(current_title_lines)
            }
        )
    # 2.9 返回结果
    return chunks_title

def improve_chunk_size(chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
    logger.info(f"没有切割，也没有合并的原始数据长度：{len(chunks)}")
    # 3.1 准备工作,定义接收的chunks,list[dict[str,Any]]
    final_chunks:list[dict[str,Any]]  =  []
    # 3.2 chunks循环,获取每个chunk,content先判断长度是否超过600,超过,进行递归切割器切割 [_函数]
    # 条件: content > 600
    for chunk in chunks:
        if len(chunk.get("content")) > 600:
            # 切割
            final_chunks.extend(_contnent_split(chunk))
        else:
            # 直接使用
            final_chunks.append(chunk)
    logger.info(f"按照标题切割，但是没有合并的数据长度：{len(final_chunks)}")
    # 3.3 合并同一个标题下过短的chunk内容，content小于400
    final_chunks = _content_merge(final_chunks)
    logger.info(f"按标题切割，也进行了合并的数据长度：{len(final_chunks)}")
    return final_chunks


def _contnent_split(chunk:dict[str,Any])  ->list[dict[str,Any]]:
    current_title_chunks = list[dict[str,Any]] = []
    # 1.数据前置处理
    content:str = chunk.get("content")
    title  :str = chunk.get("title")
    # 标题\n
    content     = content[len(title)+1:]
    pre_fix     = title+"\n"
    # 2.声明递归切割器
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","。","!","?",";"],
        chunk_size=CHUNK_SIZE - len(pre_fix) ,
        chunk_overlap=CHUNK_OVERLAP
    )
    # 3.递归切割器切割数据
    chunk_list:list[str] = text_splitter.split_text(content)
    for index,chunk_content in enumerate(chunk_list,start=1):
        sub_chunk = {
            "parent_title":title,
            "title":f"{title}_{index}",
            "part":index,
            "content":pre_fix+chunk_content
        }
        current_title_chunks.append(sub_chunk)
    logger.info(f"已经完成{title}内容切割！切割的数量:{len(current_title_chunks)}")
    return current_title_chunks

    # 3.3 合并同一个标题下过短的chunk内容,content小于400 [_函数]
    # 条件: 1. 同一个标题 2. content小于400 3. 不能大于1000
    pass
    # 3.4 返回处理长度后的chunks




def _content_merge(final_chunks:list[dict[str,Any]])  ->list[dict[str,Any]]:
    chunks: list[dict[str, Any]] = []  # 装最终合并的数据
    base_chunk: dict[str, Any] = None  # 第一个值给base
    logger.info(f"合并的数据长度:{len(final_chunks)}")
    # 进行数据循环处理
    for next_chunk in final_chunks:
        if not base_chunk:
            # 第一次循环, next->base
            base_chunk = next_chunk
            continue
        # 第二次循环开始 base有值 next是否能够合并即可
        # (后续代码被截断了)
        if len(base_chunk.get("content")) < CHUNK_MIN:
            # 小于400
            # 判断是不是同一个父标题！天坑：必须添加非空校验，否则会将两个独立的标题合并到一起！
            if base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get("parent_title"):
                # 判断合并后的长度是否小于1000
                if (len(base_chunk.get("content")) + len(next_chunk.get("content"))) < CHUNK_MAX_SIZE:
                    # 真的可以合并
                    # part 1 4 6 7  -> 1 2 3 4
                    base_chunk['content'] = base_chunk.get("content") + "\n" + next_chunk.get("content")
                else:
                    chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                chunks.append(base_chunk)
                base_chunk = next_chunk
        else:
            chunks.append(base_chunk)
            base_chunk = next_chunk

    # 跳出循环
    if base_chunk:
        # 最后一个一定没有添加进来！
        chunks.append(base_chunk)
    logger.info(f"合并之后的数据长度：{len(chunks)}")
    return chunks

def append_chunk_meta_data(chunks:list[dict[str,Any]],file_title:str):
    # 4.1 循环chunks
    for chunk in chunks:
        chunk['file_title'] = file_title
        if "part" not in chunk:
            chunk['part'] = 1
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk['title']
    logger.info(f"完成所有chunk的元数据补充!")


def backup_chunks_json(chunks, md_path_obj: Path):
    # 5.1 chunks.json文件的Path对象 md_path_obj.parent / "chunks.json"
    chunks_json_obj: Path = md_path_obj.parent / "chunks.json"

    # 5.2 对象.write_text( json.dumps(chunks),"utf-8" )
    chunks_json_obj.write_text(data=json.dumps(chunks, indent=4, ensure_ascii=False), encoding="utf-8")
    """
        json
            dump  python -> json  [备份]
                dumps  python -> json字符串 -> 我们 -> 后续
                dump   python -> json字符串 -> 写到本地磁盘的位置
            load  json   -> python [还原]
    """
    # 5.2 对象.write_text( json.dumps(chunks),"utf-8" )
    # with chunks_json_obj.open(mode="w", encoding="utf-8") as f:
    #     json.dump(chunks,f,indent=4,ensure_ascii=False)

    logger.info(f"已经完成{md_path_obj.name}对应切换内容的数据备份!")




def split_document(state: ImportGraphState) -> ImportGraphState:
    # 1.参数校验和数据获取
    md_content,file_title,md_path_obj = validates_and_get_data(state)
    # 2.根据语义进行初步切割
    title_chunks:list[dict[str,Any]] = document_split_by_title(md_content)
    # 3. 进行同一个标题下的长度优化处理和添加重叠部分
    chunks: list[dict[str, Any]] = improve_chunk_size(title_chunks)
    # 4. 元数据的补充和完善
    append_chunk_meta_data(chunks, file_title)
    # 5. 数据备份
    backup_chunks_json(chunks, md_path_obj)
    # 6. 更新state
    state['chunks'] = chunks
    # 7. 返回结果
    return state
