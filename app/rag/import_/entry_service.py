from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger
from pathlib  import Path
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    #  1.先从state读取使用的数据 local_file_path  注意：别要直接[]避免KeyError  使用get
    local_file_path:str = state.get("local_file_path")
    #  2.local_file_path_做非空判断  注意：必须判断，不要过度！
    if not local_file_path:
        # 为空 local_file_path error + raise
        #  前因 后果 核心参数 佐证
        logger.error(f"local_file_path为空，业务无法继续进行，提前终止！！")
        raise ValueError(f"local_file_path为空，业务无法继续进行，提前终止")
    #  3.不为空进行后缀名判断  md / pdf ->修改对应的值 else 既不是md又不是pdf 打印日志
    if local_file_path.lower().endswith(".md"):
        #字典名['键名'] = 新的值
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.lower().endswith(".pdf"):
        #字典名['键名'] = 新的值
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    else:
        return state

    #  4.将地址转成Path对象 （os.path）,快速获取文件file_title
    local_file_obj:Path = Path(local_file_path)
    file_title:str      = local_file_obj.stem
    """
    Path使用分三步：
     1. 创建对应的Path对象
        x:Path = Path( 字符串地址 ) 获取地址对应的Path对象
        x:Path = Path( Path对象 )  暂时用处不大！！
        x:Path = Path()           获取当前文件所在的文件夹  类似于linux的pwd
        x:Path = Path(__file__)   获取当前文件的Path地址
        x:Path = path对象 / “字符串地址” / “字符串地址”      
     2.  .属性
         .name   文件的完整文件名   info.pdf
         .stem   获取文件名部分     info
         .suffix 获取后缀名         .pdf
         .parent 获取父Path对象    Path(__file__).parent  这是上一级
         .parents 获取所有的父文件夹 Path(__file__).parents
     3.  .方法()  创建一个文件夹   查询文件是否存在
          is_file()  is_dir()  exists()   mk_dir()   read_bytes()   read_text()   write_bytes()   write_text()   iterdir()   glob()   rglob()
    """
    #  5.state 属性更新
    state["file_title"] = file_title

    #  6.返回state
    return state