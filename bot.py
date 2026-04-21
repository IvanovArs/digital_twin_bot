import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes
)
import sys
import io
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log_level
)
logger = logging.getLogger(__name__)
BOT_TOKEN = os.getenv('BOT_TOKEN')
TEACHER_TELEGRAM_ID = int(os.getenv('TEACHER_TELEGRAM_ID', 0))
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'digital_twin'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )
def save_user(telegram_id, full_name, username):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (telegram_id, full_name, role, created_at)
            VALUES (%s, %s, 'student', %s)
            ON CONFLICT (telegram_id) DO NOTHING
        """, (telegram_id, full_name or username or str(telegram_id), datetime.now()))
        conn.commit()
        logger.info(f"User saved: {telegram_id} - {full_name}")
    except Exception as e:
        logger.error(f"Error saving user: {e}")
    finally:
        cur.close()
        conn.close()
def save_chat_history(telegram_id, message, response):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, message, response, timestamp)
            VALUES ((SELECT user_id FROM users WHERE telegram_id = %s), %s, %s, %s)
        """, (telegram_id, message, response, datetime.now()))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
    finally:
        cur.close()
        conn.close()
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    save_user(user.id, user.full_name, user.username)
    
    welcome_text = """
🎓 Добро пожаловать в бота "Цифровой двойник преподавателя"!

Я помогаю студентам с вопросами по курсу 
"Основы теории информационных систем".

📌 Доступные команды:
/help - список всех команд
/glossary - глоссарий терминов
/faq - частые вопросы
/deadlines - дедлайны по работам
/consult - запись на консультацию
/my_consults - мои консультации
/feedback - оставить отзыв
/stats - моя статистика

Чем могу помочь?
"""
    await update.message.reply_text(welcome_text)
    save_chat_history(user.id, "/start", welcome_text)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
📚 *Справка по командам бота*
🔹 `/start` - Начало работы
🔹 `/glossary` - Поиск терминов в глоссарии
🔹 `/faq` - Часто задаваемые вопросы
🔹 `/deadlines` - Список дедлайнов
🔹 `/consult` - Запись на консультацию
🔹 `/my_consults` - Мои консультации
🔹 `/feedback` - Оставить отзыв
🔹 `/stats` - Моя статистика
🔹 `/help` - Эта справка

*Примеры вопросов:*
• "Что такое системно-целевой подход?"
• "Когда сдавать часть 1?"
• "Как записаться на консультацию?"
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')
async def glossary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT term, definition FROM glossary ORDER BY term LIMIT 15")
    terms = cur.fetchall()
    cur.close()
    conn.close()
    if not terms:
        await update.message.reply_text("📖 Глоссарий пока пуст. Обратитесь к преподавателю.")
        return
    text = "📖 *Глоссарий терминов:*\n\n"
    for t in terms:
        definition = t['definition'][:100] + "..." if len(t['definition']) > 100 else t['definition']
        text += f"• *{t['term']}*: {definition}\n\n"
    text += "\n💡 *Совет:* Введите название любого термина для подробного объяснения."
    await update.message.reply_text(text, parse_mode='Markdown')
async def search_term(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.lower()
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT term, definition FROM glossary 
        WHERE LOWER(term) LIKE %s OR LOWER(definition) LIKE %s
        LIMIT 3
    """, (f'%{query}%', f'%{query}%'))
    results = cur.fetchall()
    cur.close()
    conn.close()
    if results:
        text = f"🔍 *Результаты поиска по запросу*: \"{query}\"\n\n"
        for r in results:
            text += f"📌 *{r['term']}*\n{r['definition']}\n\n"
        save_chat_history(update.effective_user.id, query, text[:500])
    else:
        text = f"❌ Ничего не найдено по запросу \"{query}\".\n\nПопробуйте:\n• /glossary - посмотреть все термины\n• /faq - частые вопросы"
    await update.message.reply_text(text, parse_mode='Markdown')
async def faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT question, answer FROM faq LIMIT 10")
    faqs = cur.fetchall()
    cur.close()
    conn.close()
    if not faqs:
        await update.message.reply_text("❓ FAQ пока пуст. Вопросы можно задать преподавателю лично.")
        return
    text = "❓ *Часто задаваемые вопросы:*\n\n"
    for i, f in enumerate(faqs, 1):
        answer = f['answer'][:150] + "..." if len(f['answer']) > 150 else f['answer']
        text += f"*{i}. {f['question']}*\n{answer}\n\n"
    await update.message.reply_text(text, parse_mode='Markdown')
async def deadlines_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT subject, description, due_date, group_name 
        FROM deadlines 
        WHERE due_date >= CURRENT_DATE
        ORDER BY due_date
        LIMIT 15
    """)
    deadlines = cur.fetchall()
    cur.close()
    conn.close()
    if not deadlines:
        await update.message.reply_text("📭 Активных дедлайнов пока нет. Отдыхайте! 🎉")
        return
    text = "⏰ *Активные дедлайны:*\n\n"
    for d in deadlines:
        due = d['due_date'].strftime('%d.%m.%Y')
        group = d['group_name'] if d['group_name'] else "все группы"
        text += f"📚 *{d['subject']}*\n{d['description']}\n📅 до {due}\n👥 {group}\n\n"
    await update.message.reply_text(text, parse_mode='Markdown')
async def consult_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📅 10.04 (чт) 14:00", callback_data="consult_2026-04-10 14:00:00")],
        [InlineKeyboardButton("📅 15.04 (вт) 13:00", callback_data="consult_2026-04-15 13:00:00")],
        [InlineKeyboardButton("📅 16.04 (ср) 10:00", callback_data="consult_2026-04-16 10:00:00")],
        [InlineKeyboardButton("📅 22.04 (вт) 14:00", callback_data="consult_2026-04-22 14:00:00")],
        [InlineKeyboardButton("📅 29.04 (вт) 11:00", callback_data="consult_2026-04-29 11:00:00")],
        [InlineKeyboardButton("❌ Отмена", callback_data="consult_cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📅 *Выберите удобное время для консультации:*\n\n"
        "После записи вы получите подтверждение.",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
async def consult_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = query.from_user
    data = query.data
    if data == "consult_cancel":
        await query.edit_message_text("❌ Запись на консультацию отменена.")
        return
    datetime_str = data.replace("consult_", "")
    consult_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM users WHERE telegram_id = %s", (user.id,))
        user_result = cur.fetchone()
        if user_result:
            user_id = user_result[0]
            cur.execute("""
                INSERT INTO consultations (teacher_id, student_id, datetime, status, topic)
                VALUES (%s, %s, %s, 'pending', 'Консультация по курсу "Основы ТИС"')
            """, (1, user_id, consult_datetime))
            conn.commit()
            date_str = consult_datetime.strftime('%d.%m.%Y в %H:%M')
            await query.edit_message_text(
                f"✅ *Вы записаны на консультацию*\n\n"
                f"📅 Дата: {date_str}\n"
                f"👨‍🏫 Преподаватель: Логинова А.В.\n\n"
                f"Ожидайте подтверждения. При необходимости отмены свяжитесь с преподавателем.",
                parse_mode='Markdown'
            )
            if TEACHER_TELEGRAM_ID:
                try:
                    await context.bot.send_message(
                        TEACHER_TELEGRAM_ID,
                        f"📅 *Новая запись на консультацию!*\n\n"
                        f"Студент: {user.full_name}\n"
                        f"Время: {consult_datetime.strftime('%d.%m.%Y %H:%M')}\n"
                        f"Telegram: @{user.username}" if user.username else f"ID: {user.id}",
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logger.error(f"Failed to notify teacher: {e}")
        else:
            await query.edit_message_text("❌ Ошибка: пользователь не найден. Попробуйте /start")
    except Exception as e:
        logger.error(f"Consult error: {e}")
        await query.edit_message_text("❌ Ошибка при записи. Попробуйте позже или свяжитесь с преподавателем.")
    finally:
        cur.close()
        conn.close()
async def my_consults_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT c.datetime, c.status, c.topic
        FROM consultations c
        WHERE c.student_id = (SELECT user_id FROM users WHERE telegram_id = %s)
        ORDER BY c.datetime DESC
    """, (user.id,))
    consults = cur.fetchall()
    cur.close()
    conn.close()
    if not consults:
        await update.message.reply_text("📋 У вас пока нет записей на консультации.")
        return
    status_emoji = {
        "pending": "⏳ ожидает",
        "confirmed": "✅ подтверждена",
        "cancelled": "❌ отменена",
        "completed": "✔️ завершена"
    }
    text = "📋 *Мои консультации:*\n\n"
    for c in consults:
        date_str = c['datetime'].strftime('%d.%m.%Y %H:%M')
        status = status_emoji.get(c['status'], c['status'])
        text += f"📅 {date_str}\n📝 {c['topic']}\n🔘 {status}\n\n"
    await update.message.reply_text(text, parse_mode='Markdown')
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📝 *Оставьте отзыв о работе бота*\n\n"
        "Напишите в формате:\n"
        "`Оценка: 5`\n"
        "`Текст вашего отзыва...`\n\n"
        "Оценка от 1 до 5, где 5 — отлично.",
        parse_mode='Markdown'
    )
    context.user_data['awaiting_feedback'] = True
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM consultations c
        JOIN users u ON c.student_id = u.user_id
        WHERE u.telegram_id = %s
    """, (user.id,))
    consult_count = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM chat_history ch
        JOIN users u ON ch.user_id = u.user_id
        WHERE u.telegram_id = %s
    """, (user.id,))
    message_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    text = f"""
📊 *Ваша статистика использования бота:*
💬 Сообщений боту: {message_count}
📅 Записей на консультации: {consult_count}
🎓 Роль: Студент
---
Бот помогает вам учиться? Оставьте отзыв командой /feedback
"""
    await update.message.reply_text(text, parse_mode='Markdown')
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user = update.effective_user
    message_text = update.message.text
    if context.user_data.get('awaiting_feedback'):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            rating_match = re.search(r'Оценка:\s*(\d+)', message_text)
            rating = int(rating_match.group(1)) if rating_match else None
            comment = re.sub(r'Оценка:\s*\d+\s*', '', message_text).strip()
            if rating and 1 <= rating <= 5:
                cur.execute("""
                    INSERT INTO user_feedback (user_id, rating, comment, created_at)
                    VALUES ((SELECT user_id FROM users WHERE telegram_id = %s), %s, %s, %s)
                """, (user.id, rating, comment or "Без комментария", datetime.now()))
                conn.commit()
                await update.message.reply_text("🙏 Спасибо за ваш отзыв!", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Укажите оценку от 1 до 5 в формате: `Оценка: 5`", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Feedback error: {e}")
            await update.message.reply_text("❌ Ошибка при сохранении отзыва.")
        finally:
            cur.close()
            conn.close()
        context.user_data['awaiting_feedback'] = False
        return
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT term, definition FROM glossary 
        WHERE LOWER(term) LIKE %s OR LOWER(definition) LIKE %s
        LIMIT 3
    """, (f'%{message_text.lower()}%', f'%{message_text.lower()}%'))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    if results:
        text = f"🔍 *По запросу*: \"{message_text}\"\n\n"
        for r in results:
            text += f"📖 *{r['term']}*\n{r['definition']}\n\n"
        save_chat_history(user.id, message_text, text[:500])
        await update.message.reply_text(text, parse_mode='Markdown')
    else:
        await update.message.reply_text(
            f"❌ Ничего не найдено по запросу \"{message_text}\".\n\n"
            f"💡 Попробуйте:\n"
            f"• /glossary - посмотреть все термины\n"
            f"• /faq - частые вопросы\n"
            f"• /consult - записаться на консультацию",
            parse_mode='Markdown'
        )
        save_chat_history(user.id, message_text, "No results found")
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found in .env file")
        print("[ERROR] BOT_TOKEN not found in .env file")
        return
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("glossary", glossary_command))
    app.add_handler(CommandHandler("faq", faq_command))
    app.add_handler(CommandHandler("deadlines", deadlines_command))
    app.add_handler(CommandHandler("consult", consult_command))
    app.add_handler(CommandHandler("my_consults", my_consults_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CallbackQueryHandler(consult_callback, pattern="consult_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен...")
    print("[INFO] Bot 'Digital Twin of the Teacher' started!")
    print(f"[INFO] Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print("[INFO] Press Ctrl+C to stop")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == "__main__":
    main()
