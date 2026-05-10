from PIL import Image
import base64
from io import BytesIO

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def pil_image_to_base64(img, format='PNG'):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def get_image_dimensions_from_base64(base64_string):
    """
    Get image width and height from base64 encoded image data.
    
    Args:
        base64_string (str): Base64 encoded image data
        
    Returns:
        tuple: (width, height) of the image
    """
    # Decode base64 string to bytes
    image_data = base64.b64decode(base64_string)
    # Create image from bytes
    image = Image.open(BytesIO(image_data))
    # Get dimensions
    width, height = image.size
    return width, height


def resize_image(image, max_size=448, min_size=224):
    """
    Resize a PIL image while maintaining aspect ratio. 
    The smallest side is resized to min_size if the longest side exceeds max_size.
    """
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        if min(new_width, new_height) < min_size:
            if new_width < new_height:
                new_width = min_size
                new_height = int(height * (min_size / width))
            else:
                new_height = min_size
                new_width = int(width * (min_size / height))

        return image.resize((new_width, new_height), Image.LANCZOS)
    return image