import time

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.shared.runtime.logger import logger , PROJECT_ROOT
from app.infra.config.providers import infra_config  ##infra_config是providers中的对象名
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS
import requests

def validates_pdf_paths(state:ImportGraphState)  -> tuple[Path,Path]:
    # 3.1 从state中获取pdf_path和local_dir地址  ->str。pdf_path指的是某个具体pdf文件的路径，lcoal_dir指的是本地的一个文件夹的路径
    pdf_path:str  =  state.get("pdf_path")
    local_dir:str  =  state.get("local_dir")
    # 3.2 非空校验
    if not pdf_path:  #pdf文件路径
        #前因  后果  关键参数
        logger.error(f"pdf_path地址为空！业务无法继续，提前终止！")
        raise ValueError(f"pdf_path为空！业务无法继续，提前终止！")
    if not local_dir:  #本地输出目录
        local_dir = PROJECT_ROOT / "output"
        logger.warning(f"local_dir没有赋予默认值，配置默认输出地址为:{local_dir},业务继续进行")
        state["local_dir"] = str(local_dir)
    # 3.3 将pdf_path和local_dir转成Path对象，方便后续操作  -> Path
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 3.4 检查pdf_path_obj 是否真实存在 exist（）
    if not pdf_path_obj.is_file():
       logger.error(f"pdf_path路径地址为：{pdf_path},但是检查没有真实对应到文件！业务无法继续，提前终止！！")
       raise FileNotFoundError(f"pdf_path路径地址为{pdf_path},但是检查没有真实对应到文件！业务无法继续，提前终止！！")
    # 3.5 检查local_dir_obj是否真实存在  is_dir() !
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir路径为:{local_dir},但是没有具体的文件夹，直接创建对应的文件夹，业务继续进行！！")
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    # 3.6 return pdf_path_obj ,local_dir_obj
    return pdf_path_obj,local_dir_obj

def apply_mineru_get_upload_url(pdf_file_name:str)  ->tuple[str,str]:
    # 4.1 组装对应的请求头（token）
    token = infra_config.mineru.api_key
    header = {
        "Content-Type":"application/json",
        "Authorization":f"Bearer {token}"
    }
    #  4.2 组装对应的请求体json数据（请求参数）
    data = {
        "files":[
            {"name":pdf_file_name}
        ],
        "model_version":MINERU_MODEL_VERSION,
        "extra_formats":["html"]
    }
    # 4.3 向对应的接口发起post方式请求，并等待获得结果
    url = f"{infra_config.mineru.base_url}/file-urls/batch"
    response = requests.post(
       url = url ,
       headers = header ,
       json = data ,
       timeout = MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )
   #4.4 判断http协议网络状态status_code
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"向mineru服务器申请解析地址，但是出现网络异常！状态码为:{status_code},业务无法继续，提前终止")
        raise RuntimeError(f"向mineru服务器申请解析地址，但是出现网络异常！状态码为:{status_code},业务无法继续，提前终止")
    # 4.5 判断业务状态码code
    response_str = response.json()
    code = response_str.get("code",-1)   ##没有code的话，值为-1
    if code != 0:
        logger.error(f"向mineru服务器申请解析地址，但是出现网络异常！状态码为:{code},业务无法继续，提前终止")
        raise RuntimeError(f"向mineru服务器申请解析地址，但是出现网络异常！状态码为:{code},业务无法继续，提前终止")
    # 4.6 获取目标结果batch_id/upload_url地址
    json_data = response_str.get("data",{})
    batch_id:str = json_data.get("batch_id")
    upload_file_url:str = json_data.get("file_urls")[0]
    # 4.7 返回结果即可
    return batch_id,upload_file_url        #batch_id（批次 ID）就是这次“批量上传任务”的唯一凭证

def upload_file_to_url(upload_url:str,pdf_path_obj:Path)  ->None:
    """
    5.1  读取上传文件的数据 pdf_path_obj ->字节的数据
    5.2 向指定的地址发起put请求，传递pdf文件数据
    5.3 判断响应状态码，不是200抛出异常！[本次只需要判断http状态码，不需要判断业务code!本身就是不是mineru业务接口，文件服务器的上传地址]
    """
    # 5.1 读取上传文件的数据 pdf_path_obj ->字节的数据
    data = pdf_path_obj.read_bytes()
    #5.2 向指定的地址发起put请求，传递pdf文件数据
    #response = requests.put(url=upload_url,data=data,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    #request.put->一次请求;session->1.复用请求2.session.trust_env=Falise 不要给我额外的添加任何请求！我本次设置了什么，就给我传递什么
    #下面那行的意思是：我要创建一个网络请求回话（requests.Session()），我管他叫session，接下来我要用session去发请求，用完以后，自动帮我把他关掉
    with requests.Session() as session:
        session.trust_env = False
        response = session.put(url=upload_url,data=data,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)

    #5.3 判断响应状态码，不是200抛出异常！[本次只需要判断http状态码，不需要判断业务code!
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"向:{upload_url}地址上传文件，网络状态失败：{status_code},业务无法继续，提前终止！！")
        raise RuntimeError(f"向:{upload_url}地址上传文件，网络状态失败：{status_code},业务无法继续，提前终止！！")
def poll_mineru_zip_url(batch_id:str)  ->str:
    # 6.1 接收变量参数 临时变量接收值 最大轮询时间 / 轮询间隔时间 / 记录起始时间
    max_timeout  = MINERU_DOWNLOAD_TIMEOUT_SECONDS
    interval_time= MINERU_DOWNLOAD_TIMEOUT_SECONDS
    start_time   = time.time()  #时间秒

    url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
    header = {
        "Content-Type":"application/json",
        "Authorization":f"Bearer{infra_config.mineru_config.api_key}"
    }
    # 6.2 死循环 while True:
    while True:
        # 6.2.1 有没有超过最大等待时间 []
        if time.time() - start_time >= max_timeout:
            logger.error(f"轮询获取解析结果时间超时！等待时间为{time.time()-start_time },最大等待时间:{max_timeout},业务无法继续，提前终止！！")
            raise TimeoutError(f"轮询获取解析结果时间超时！等待时间为{time.time()-start_time },最大等待时间:{max_timeout},业务无法继续，提前终止！！")
        # 6.2.2 向指定的地址请求查询解析状态
        response = requests.get(url,headers=header,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
        #
        # 6.2.3 判断网络状态码(HTTP状态码)status_code != 200  [判断是否可以给与机会(5##)，可以等待重试 / 其他的也需要抛出异常终止轮询]
        status_code = response.status_code
        if status_code != 200:
            #200代表网络请求成功，5开头代表请求成功，但是对面的服务器处理时发生问题
            if 500 <= status_code <600:
                logger.warning(f"轮询获取解析结果，出现服务器异常,状态码为:{status_code},稍后重试！！")
                time.sleep(interval_time)
                continue
            #错误不可以重试
            logger.error(
                f"轮询获取解析结果，出现异常，状态码为:{status_code},异常是不可修复和重试，业务无法继续，提前终止！！")
            raise RuntimeError(
                f"轮询获取解析结果，出现异常，状态码为:{status_code},异常是不可修复和重试，业务无法继续，提前终止！！")
        # 6.2.4 判断业务状态码code != 0 [需要抛出异常终止轮询]
        json_data = response.json()
        code = json_data.get("code",-1)
        if code != 0:
            logger.error(f"轮询获取解析结果，业务状态出现异常，code为{code},异常是不可修复和重试，业务无法继续，提前终止！！")
            raise RuntimeError(f"轮询获取解析结果，业务状态出现异常，code为{code},异常是不可修复和重试，业务无法继续，提前终止！！")
        # 判断文件的解析状态state
        result = json_data.get("data",{}).get("extract_result",[])[0]
        state = result.get("state")

        if state == "done":
            #解析完成了
            logger.info(f"文件解析完成！！")
            full_zip_url:str = result.get("full_zip_url")
            return full_zip_url
        elif state == "failed":
            # 解析失败了
            logger.error(f"文件解析失败，业务无法继续，提前终止！！")
            raise RuntimeError(f"文件解析失败，业务无法继续，提前终止！")
        else:
            #正在解析中
            logger.info(f"文件正在解析中，当前状态为{state},稍后重试")

def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:

    # 1. 进行参数合理性校验
    pdf_path_obj,local_dir_obj = validates_pdf_paths(state)
    # 2.向mineru申请文件上传地址和batch_id
    batch_id,upload_file_url = apply_mineru_get_upload_url(pdf_path_obj.name)
    # 3.向指定的文件地址（预签名地址）上传文件数据
    upload_file_to_url(upload_file_url,pdf_path_obj)
    # 4.轮询获取解析结果
    full_zip_url:str = poll_mineru_zip_url(batch_id)
    return state
 # code 是业务逻辑状态码，MinerU 服务应用层定义的返回码。而 status_code 通常是网络协议层（HTTP）的状态码。status_code 告诉你 “请求是否成功到达了服务器”，而 code 告诉你 “服务器处理这个请求的结果是什么”。
