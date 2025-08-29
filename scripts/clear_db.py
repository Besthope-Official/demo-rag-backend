from loguru import logger
from pymongo import MongoClient

# 配置日志
logger.add("clear_user_db.log", format="{time} {level} {message}", level="INFO", rotation="1 MB")

# MongoDB 连接配置
MONGO_URL = "mongodb://admin:A2YaLSspviFkaZ3e@139.155.150.18:27017"
DATABASE_NAME = "default"


def clear_database():
    try:
        # 连接 MongoDB
        client = MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]

        # 获取所有集合名称
        collections = db.list_collection_names()
        logger.info(f"找到集合: {collections}")

        # 清空每个集合
        for collection_name in collections:
            result = db[collection_name].delete_many({})
            logger.info(f"清空集合 {collection_name}, 删除记录数: {result.deleted_count}")

        # 关闭连接
        client.close()
        logger.info(f"数据库 {DATABASE_NAME} 清空完成")
    except Exception as e:
        logger.error(f"清空数据库失败: {e}")
        raise


if __name__ == "__main__":
    clear_database()
