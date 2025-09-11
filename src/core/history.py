"""管理会话历史"""

from datetime import datetime

from loguru import logger

from ..schema import ChatHistory, ChatSummary, Message
from .user import UserDAO


class HistoryService:
    """会话历史服务，使用 MongoDB 持久化。"""

    def __init__(self, dao: UserDAO):
        self.dao = dao
        self.dao.db["chats"].create_index([("username", 1), ("chat_id", 1)])  # 添加索引

    async def save_chat_history(
        self, username: str, chat_id: str, messages: list[Message], title: str
    ) -> None:
        """保存聊天历史到 MongoDB。"""
        try:
            collection = self.dao.db["chats"]
            collection.update_one(
                {"username": username, "chat_id": chat_id},
                {
                    "$set": {
                        "username": username,
                        "chat_id": chat_id,
                        "title": title,
                        "messages": [msg.model_dump() for msg in messages],
                        "created_at": datetime.now().timestamp(),
                        "updated_at": datetime.now().timestamp(),
                    }
                },
                upsert=True,
            )
            logger.info(f"保存聊天历史: username={username}, chat_id={chat_id}, title={title}")
        except Exception as e:
            logger.error(f"保存聊天 {chat_id} 历史失败: {e}")

    async def get_chat_list(self, username: str) -> list[ChatSummary]:
        """获取用户所有聊天列表。"""
        try:
            collection = self.dao.db["chats"]
            chats = list(collection.find({"username": username}))
            summaries = [
                ChatSummary(
                    chat_id=chat["chat_id"],
                    title=chat.get("title", "无标题"),
                    created_at=chat.get("created_at", 0),
                    updated_at=chat.get("updated_at", 0),
                )
                for chat in chats
            ]
            logger.info(f"获取聊天列表: username={username}, count={len(summaries)}")
            return summaries
        except Exception as e:
            logger.error(f"获取用户 {username} 聊天列表失败: {e}")
            return []

    async def get_chat_history(self, chat_id: str) -> ChatHistory | None:
        """获取指定 chat_id 的完整历史。"""
        try:
            collection = self.dao.db["chats"]
            chat = collection.find_one({"chat_id": chat_id})
            if chat:
                return ChatHistory(**chat)
            logger.warning(f"聊天 {chat_id} 不存在")
            return None
        except Exception as e:
            logger.error(f"获取聊天 {chat_id} 历史失败: {e}")
            return None
