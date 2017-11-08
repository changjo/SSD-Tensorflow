## For Displaying Korean
from PIL import Image, ImageDraw, ImageFont

font_size = 36
font_color = (0, 0, 0)
caption = u'안녕하세요'
unicode_font = ImageFont.truetype("NanumGothic.ttf", font_size)

pil_image = Image.open(path + image_names[-5]).convert('RGB')
draw = ImageDraw.Draw(pil_image)
draw.text ( (10,10), caption, font=unicode_font, fill=font_color )
open_cv_image = np.array(pil_image)
open_cv_image = open_cv_image[:, :, ::-1].copy()