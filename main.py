
from PIL import Image
import model
import telebot
from telebot import types
from settings import token

steps = 300
bot = telebot.TeleBot(token)


def create_menu(message):
    btn1 = types.KeyboardButton('Загрузить картинку для изменения')
    btn2 = types.KeyboardButton('Загрузить картинку со стилем')
    btn3 = types.KeyboardButton('Начать работу')

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id, 'Выберите действие', reply_markup=markup)


@bot.message_handler(commands=['start'])
def start_message(message):
    StartAnswer = "Привет " + message.from_user.first_name
    bot.send_message(message.chat.id, StartAnswer)
    create_menu(message)


@bot.message_handler(content_types=['photo'])
def mess(message):
    if (message.chat.id not in last_image) or (last_image[message.chat.id]) == 0:
        orig_image[message.chat.id] = message.photo[0].file_id
    else:
        style_image[message.chat.id] = message.photo[0].file_id

    # print(message.photo[0].file_id)
    # f = bot.get_file(orig_image[message.chat.id])
    # print(f)


@bot.message_handler(content_types=['text'])
def mess(message):
    get_message_bot = message.text.strip()

    if get_message_bot == 'Загрузить картинку для изменения':
        bot.send_message(message.chat.id, 'Загрузите картинку для изменения')
        last_image[message.chat.id] = 0
    elif get_message_bot == 'Загрузить картинку со стилем':
        bot.send_message(message.chat.id, 'Загрузите картинку со стилем')
        last_image[message.chat.id] = 1
    elif get_message_bot == 'Начать работу':

        if message.chat.id not in orig_image.keys():
            bot.send_message(message.chat.id, 'Не загружена картинка для изменений')
        if message.chat.id not in style_image.keys():
            bot.send_message(message.chat.id, 'Не загружена картинка со стилем')

        if message.chat.id in orig_image.keys() and message.chat.id in style_image.keys():
            # idphoto = style_image[message.chat.id]

            f = open("demofile3.txt", "w")
            f.write("Woops! I have deleted the content!")
            f.close()

            #
            orig_name = 'orig.jpg'
            style_name = 'style.jpg'
            target_name = 'res.png'

            with open(orig_name, 'wb') as new_file:
                file_info = bot.get_file(orig_image[message.chat.id])
                down_file = bot.download_file(file_info.file_path)
                new_file.write(down_file)

            with open(style_name, 'wb') as new_file:
                file_info = bot.get_file(style_image[message.chat.id])
                down_file = bot.download_file(file_info.file_path)
                new_file.write(down_file)

            bot.send_message(message.chat.id, 'Начинается преобразование картинки. Это займет некоторое время.')

            # print('Loading images')
            content = model.load_image(orig_name).to(device)
            style = model.load_image(style_name, shape=content.shape[-2:]).to(device)
            target = model.transfer_style(device, vgg, content, style, steps)
            target = model.im_convert(target) * 255
            #
            img = Image.fromarray(target.astype('uint8'), mode="RGB")
            img.save(target_name)

            img = open(target_name, 'rb')
            bot.send_photo(message.chat.id, img)


if __name__ == '__main__':
    device = model.init_device()
    print('Loading model')
    vgg = model.init_model(device)


    last_image = {}
    orig_image = {}
    style_image = {}
    target_image = {}

    print('Start pooling')
    bot.infinity_polling()
