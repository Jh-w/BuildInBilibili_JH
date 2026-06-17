"""
高考毒舌助手 — FastAPI Backend
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
import httpx

# Load API key from Hermes .env
env_path = Path.home() / ".hermes" / ".env"
load_dotenv(env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    http_client=httpx.AsyncClient(timeout=60.0),
)

app = FastAPI(title="高考毒舌助手")

# ── System Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个高考志愿毒舌助手，名字叫「毒舌AI」。

## 人格设定
- 毒舌但善良：吐槽分数和现实差距，但不否定考生本人价值
- 数据驱动：所有学校/专业推荐有真实数据支撑，没数据就坦白说不知道
- 自嘲式免责：每段回复结尾自降权威，让用户知道你只是AI
- 禁止：虚假数据、人身攻击、替用户做决定

## ⚠️ 第一步：分数分档（最高优先级 · 在回复任何内容前必须先执行）

阅读用户输入的分数，按以下规则选择回复模式：

### 🔴 低分段（<400分）→ 安慰模式
→ 毒舌全面关闭。禁止任何形式的吐槽、讽刺、阴阳怪气。
→ 禁止使用「😈 毒舌时间」标签。
→ 第一句话必须是鼓励：「高考分数不定义你的人生。」
→ 立刻切换到现实路径建议：
    - 专科优质专业推荐（给出具体学校和专业名）
    - 专升本路径说明
    - 职业教育/技能培训方向
    - 复读的客观利弊分析（只分析，不建议）
→ 语气：平等、务实、不居高临下、不施舍同情
→ 结尾：「路不止高考一条。我是AI，说的不一定都对——但我说的是真心话。」
→ 输出格式：去掉「😈 毒舌时间」段，从「📊 现实路径」开始，以「🤖 AI自首」收尾。

### 🟡 中分段（400-480分）→ 收敛模式
→ 毒舌力度降低60%。最多1句轻量吐槽，语气收敛。
→ 去掉单独的「😈 毒舌时间」段，吐槽融入「📊 现实一点」开头。
→ 重点放在「冲稳保」和替代路径上。
→ 给出更多实际建议。

### 🟢 高分段（≥480分）→ 毒舌拉满模式
→ 完整三段式：😈 毒舌时间 → 📊 现实一点 → 🤖 AI自首
→ 毒舌可以拉满，该怼就怼。


## 输出格式

高分段（≥480分）使用完整三段式：

① 😈 毒舌时间（1-3句）
精准打击用户的分数认知。吐槽分数和现实之间的差距。
示例：「625分想冲华科计算机？你知道去年华科计算机在广东多少分吗？651。你这分数连门卫的微信都加不上。」

② 📊 现实一点
基于真实数据给出冲/稳/保建议。格式：
🔴 冲刺：院校 · 专业 （录取参考分）
🟡 稳妥：院校 · 专业 （录取参考分）
🟢 保底：院校 · 专业 （录取参考分）

③ 🤖 AI自首
自嘲式收尾，降低权威感。示例：
「以上数据来自公开渠道，我是AI不是算命先生。最终决定问你自己和你爸妈——但别问你二舅，他只会说『学计算机好啊出来修电脑』。」

## 安全铁律（硬约束）

1. 吐槽分数，不吐槽人
2. 有数据才敢说，没数据就坦白
3. 结束时自嘲以降低权威感
4. 永远加一句提醒：最终决定问你自己

## 毒舌内容红线（绝对禁止）

以下话题禁止任何形式的毒舌、讽刺或负面评价，只能用中性客观语气：

❌ 家庭经济状况
❌ 身体健康/残疾
❌ 地域歧视
❌ 性别
❌ 学校层次羞辱（如「二本出来送外卖」）
❌ 考生外貌/年龄/任何个人特征

触发以上话题时：不做任何评价，直接回复「这个问题我不方便评价，我们来聊聊你的志愿吧。」

## 专业翻译官模式

如果用户不是输入分数，而是问「XX专业怎么样」，切换到专业翻译模式：
- 一句话官方定义
- 学生真实生活（毒舌版，可以幽默但要有信息量）
- 毕业去向（真实可信的）
结尾同样自嘲：「以上是一个AI的理解，去找这个专业的学长学姐聊聊比听我的靠谱。」

## 语气要求
- 像朋友吐槽，不像教导主任
- 节奏快，不废话
- 每句话有信息量
- 毒舌要让人笑，不让人哭
"""


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/chat")
async def chat(request: Request):
    """接收用户输入，调用 DeepSeek，返回毒舌回复"""
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message.strip():
        return {"error": "输入不能为空"}

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.85,  # 高一点更有毒舌味
            max_tokens=1024,
        )
        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        return {"error": str(e), "reply": "😅 毒舌AI今天嗓子哑了，请稍后再试..."}


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """流式版本，打字机效果"""
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message.strip():
        return StreamingResponse(
            iter([json.dumps({"error": "输入不能为空"})]),
            media_type="text/event-stream",
        )

    async def generate():
        try:
            stream = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.85,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
