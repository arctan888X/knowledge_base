from app.shared.model.lm_utils import get_llm_client
from app.infra.config.providers import infra_config


class LLMProvider():

    # 大语言模型
    def chat(self, model: str | None, json_model: bool = False):
        #self 指这个 LLMProvider 类本身创建出来的对象（实例）。
        return get_llm_client(model=model, json_mode=json_model)

    # 视觉模型
    def vision_chat(self, model: str | None):
        if not model:
            model = infra_config.lm_config.lv_model
        return get_llm_client(model=model, json_mode=False)

    # ---------- 后续修改扩展 ----------
    # 嵌入式模型

    # 重排序模型


llm_provider = LLMProvider()