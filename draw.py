from turtle import width, color
from PIL import Image, ImageDraw, ImageFont
image = Image.new('RGB', (width, height), color)

draw = ImageDraw.Draw(image)
text = "HAPPY BIRTHDAY"
font = ImageFont.truetype("arial.ttf", 36)
textwidth, textheight = draw.textsize(text,font)
x = (width - textwidth) / 2
y = (height - textheight) / 2
draw.text((x, y), text, font = font, fill = (255, 255, 255))

image.save("greeting_card.png")