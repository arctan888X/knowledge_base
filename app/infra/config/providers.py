from app.shared.config.embedding_config import embedding_config
from app.shared.config.lm_config import lm_config
from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.mineru_config import mineru_config
from app.shared.config.minio_config import minio_config
from app.shared.config.reranker_config import reranker_config
from app.shared.config.settings_config import settings

from dataclasses import dataclass , field

@dataclass
class InfraConfig:
    app: object = field(default_factory=lambda: settings)
    llm: object = field(default_factory=lambda: lm_config)
    embedding: object = field(default_factory=lambda: embedding_config)
    reranker: object = field(default_factory=lambda: reranker_config)
    mcp: object = field(default_factory=lambda: mcp_config)
    milvus: object = field(default_factory=lambda: milvus_config)
    mineru: object = field(default_factory=lambda: mineru_config)
    #我在这个总配置类里声明了一个叫 mineru 的属性，如果你在创建对象时没有给它传值，就请自动去执行 lambda: mineru_config 这个函数，把它返回的结果作为默认值。
    ##: object 是类型注解；= field(...)是 dataclasses 模块提供的一个专门用来精细化控制属性的函数；
    # lambda 是 Python 的匿名函数。这句话的意思就是：“每次调用我，我就把你之前定义好的那个 mineru_config 对象拿出来给你
    minio: object = field(default_factory=lambda: minio_config)

infra_config = InfraConfig()

print(infra_config.app.import_app_name)