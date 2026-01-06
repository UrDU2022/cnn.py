import telebot
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

# ================== ENV ==================
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

# ================== DEVICE ==================
device = torch.device("cpu")

# ================== MODEL ==================
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 class: Normal / Pneumonia

model.load_state_dict(
    torch.load("pneumonia.pt", map_location=device)
)
model.eval()
model.to(device)
print("✅ Model yuklandi")

# ================== TRANSFORM ==================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ================== START ==================
@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "🤖 Salom!\n"
        "📸 Ko‘krak qafasi rentgen rasmini yuboring.\n"
        "🧠 Men pnevmoniya bor-yo‘qligini aniqlayman."
    )

# ================== IMAGE HANDLER ==================
@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}"

        response = requests.get(file_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        confidence = confidence.item() * 100

        if predicted.item() == 1:
            result = (
                "⚠️ **Pnevmoniya aniqlandi!**\n"
                f"📊 Ishonchlilik: {confidence:.2f}%\n"
                "🩺 Iltimos, shifokorga murojaat qiling."
            )
        else:
            result = (
                "✅ **Pnevmoniya aniqlanmadi**\n"
                f"📊 Ishonchlilik: {confidence:.2f}%\n"
                "💚 Sog‘ bo‘ling!"
            )

        bot.reply_to(message, result, parse_mode="Markdown")

    except Exception as e:
        print("❌ Xatolik:", e)
        bot.reply_to(
            message,
            "⚠️ Xatolik yuz berdi. Iltimos, boshqa rasm yuboring."
        )

# ================== POLLING ==================
bot.infinity_polling(timeout=10, long_polling_timeout=5)

