from PIL import Image
import numpy as np


def show_reconstructed_image(comp_n, U, S, VT):    
    # Convert to uint8 and display
    rank1_component = S[comp_n] * np.outer(U[:, comp_n], VT[comp_n, :])
    rank1_uint8 = np.clip(rank1_component, 0, 255).astype(np.uint8)
    rank1_pil = Image.fromarray(rank1_uint8)
    # display(rank1_pil.resize((400, int(rank1_pil.height * 400 / rank1_pil.width))))

    # Reconstruct the image using only the first singular value
    img_rank1 = U[:, 0:comp_n] @ np.diag(S[0:comp_n]) @ VT[0:comp_n, :]

    # Convert to uint8 and display
    img_rank1_uint8 = np.clip(img_rank1, 0, 255).astype(np.uint8)
    img_rank1_pil = Image.fromarray(img_rank1_uint8)
    # display(img_rank1_pil.resize((400, int(img_rank1_pil.height * 400 / img_rank1_pil.width))))

    # Show both images side by side for comparison
    combined_width = rank1_pil.width + img_rank1_pil.width
    combined_height = max(rank1_pil.height, img_rank1_pil.height)
    combined_image = Image.new('L', (combined_width, combined_height))
    combined_image.paste(rank1_pil, (0, 0))
    combined_image.paste(img_rank1_pil, (rank1_pil.width, 0))
    
    return combined_image.resize((800, int(combined_image.height * 800 / combined_image.width)))
        