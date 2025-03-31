import collections.abc
import numpy as np
import streamlit as st
import torch
import PIL
from PIL import Image

from model import Generator, Discriminator, transforms, DEVICE  # Импортируйте структуры моделей

collections.MutableMapping = collections.abc.MutableMapping


def main():
    # Создайте экземпляры моделей
    Gen_mtp = Generator().to(DEVICE)  # Model Monet to Photo
    Gen_ptm = Generator().to(DEVICE)  # Model Photo to Monet
    Disc_m = Discriminator().to(DEVICE)  # Discriminator for Monet
    Disc_p = Discriminator().to(DEVICE)  # Discriminator for Photo

    # Загрузка моделей
    try:
        Gen_mtp.load_state_dict(torch.load('Gen_mtp.pth', map_location=DEVICE))
        Gen_ptm.load_state_dict(torch.load('Gen_ptm.pth', map_location=DEVICE))
        Disc_m.load_state_dict(torch.load('Disc_m.pth', map_location=DEVICE))
        Disc_p.load_state_dict(torch.load('Disc_p.pth', map_location=DEVICE))
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: One or more model files not found: {e}")
    except RuntimeError as e:
        print(f"Error loading models: {e}")

    # Устанавливаем режим обучения
    # Gen_mtp.train()
    # Gen_ptm.train()
    # Disc_m.train()
    # Disc_p.train()

    # Запуск процесса обучения
    # for epoch in range(NUM_EPOCHS):
    #    print(f'=========== EPOCH №{epoch + 1} ===========')
    #    train_fn(gen_mtp, gen_ptm, disc_m, disc_p, loader, opt_gen, opt_disc, l1_loss, mse_loss, G_loss_list,
    #             D_loss_list)

    # Перевод моделей в режим оценки
    Gen_mtp.eval()
    Gen_ptm.eval()
    Disc_m.eval()
    Disc_p.eval()

    # Streamlit интерфейс
    st.title('Генератор изображений в стиле Моне')
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file).convert("RGB")  # Преобразование в формат RGB
        st.image(image, caption='Загруженное изображение', use_container_width=True)
        if st.button('Сгенерировать изображение'):
            # Преобразуйте изображение в массив NumPy
            image_np = np.array(image)

            # Примените трансформации
            image_tensor = transforms(image=image_np)['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                generated_image = Gen_ptm(image_tensor)

            # Преобразуйте результатно в изображение
            generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_image = (generated_image * 255).astype('uint8')

            # Отображение сгенерированного изображения без постобработки
            final_image = Image.fromarray(generated_image)
            st.image(final_image, caption='Сгенерированное изображение', use_container_width=True)


if __name__ == '__main__':
    main()
