import collections.abc
import numpy as np
import streamlit as st
import torch
from PIL import Image
from model_monet import Generator as Gen_ptm, Discriminator as Disc_m, loader, DEVICE, NUM_EPOCHS, transforms
from model_picasso import Generator as Gen_ptp, Discriminator as Disc_p, loader, DEVICE, NUM_EPOCHS, transforms


collections.MutableMapping = collections.abc.MutableMapping


# Определите необходимые функции, такие как train_fn, загрузка данных и т.д.

def train(Gen_ptm, Gen_ptp, Disc_m, Disc_p, loader, opt_gen_ptm, opt_gen_ptp, opt_disc_m, opt_disc_p, l1_loss,
          mse_loss):
    # Устанавливаем режим обучения
    Gen_ptm.train()
    Gen_ptp.train()
    Disc_m.train()
    Disc_p.train()

    for epoch in range(NUM_EPOCHS):
        st.write(f'=========== EPOCH №{epoch + 1} ===========')

        for batch_idx, (real_images, target_images) in enumerate(loader):
            # Переместите данные на устройство
            real_images = real_images.to(DEVICE)
            target_images = target_images.to(DEVICE)

            # Обучение дискриминаторов
            opt_disc_m.zero_grad()
            opt_disc_p.zero_grad()

            # Генерация изображений
            fake_images_ptm = Gen_ptm(real_images)
            fake_images_ptp = Gen_ptp(real_images)

            # Получение дискриминаторских предсказаний
            disc_m_loss = mse_loss(Disc_m(fake_images_ptm), torch.ones_like(Disc_m(fake_images_ptm)))
            disc_m_loss += mse_loss(Disc_m(real_images), torch.zeros_like(Disc_m(real_images)))

            disc_p_loss = mse_loss(Disc_p(fake_images_ptp), torch.ones_like(Disc_p(fake_images_ptp)))
            disc_p_loss += mse_loss(Disc_p(real_images), torch.zeros_like(Disc_p(real_images)))

            # Обратное распространение и обновление весов дискриминаторов
            disc_m_loss.backward()
            disc_p_loss.backward()
            opt_disc_m.step()
            opt_disc_p.step()

            # Обучение генераторов
            opt_gen_ptm.zero_grad()
            opt_gen_ptp.zero_grad()

            # Генерация изображений
            fake_images_ptm = Gen_ptm(real_images)
            fake_images_ptp = Gen_ptp(real_images)

            # Получение дискриминаторских предсказаний
            g_ptm_loss = mse_loss(Disc_m(fake_images_ptm), torch.ones_like(Disc_m(fake_images_ptm)))
            g_ptp_loss = mse_loss(Disc_p(fake_images_ptp), torch.ones_like(Disc_p(fake_images_ptp)))

            # Добавление L1 потерь для регуляризации
            g_ptm_loss += l1_loss(fake_images_ptm, target_images)
            g_ptp_loss += l1_loss(fake_images_ptp, target_images)

            # Обратное распространение и обновление весов генераторов
            g_ptm_loss.backward()
            g_ptp_loss.backward()
            opt_gen_ptm.step()
            opt_gen_ptp.step()

        st.write(f'Epoch {epoch + 1} завершен!')

    # Сохранение моделей после обучения
    torch.save(Gen_ptm.state_dict(), 'Gen_ptm.pth')
    torch.save(Gen_ptp.state_dict(), 'Gen_ptp.pth')
    torch.save(Disc_m.state_dict(), 'Disc_m.pth')
    torch.save(Disc_p.state_dict(), 'Disc_p.pth')
    st.write("Модели успешно сохранены!")


def main():
    # Создайте экземпляры моделей
    gen_ptm = Gen_ptm().to(DEVICE)  # Model Monet to Photo
    gen_ptp = Gen_ptp().to(DEVICE)  # Model Picasso to Photo
    disc_m = Disc_m().to(DEVICE)  # Discriminator for Monet
    disc_p = Disc_p().to(DEVICE)  # Discriminator for Photo

    # Загрузка моделей
    try:
        gen_ptm.load_state_dict(torch.load('Gen_ptm.pth', map_location=DEVICE))
        gen_ptp.load_state_dict(torch.load('Gen_ptp.pth', map_location=DEVICE))
        disc_m.load_state_dict(torch.load('Disc_m.pth', map_location=DEVICE))
        disc_p.load_state_dict(torch.load('Disc_p.pth', map_location=DEVICE))
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: One or more model files not found: {e}")
    except RuntimeError as e:
        print(f"Error loading models: {e}")

    # Определите оптимизаторы и функции потерь
    opt_gen_ptm = torch.optim.Adam(gen_ptm.parameters(), lr=0.0002)
    opt_gen_ptp = torch.optim.Adam(gen_ptp.parameters(), lr=0.0002)
    opt_disc_m = torch.optim.Adam(disc_m.parameters(), lr=0.0002)
    opt_disc_p = torch.optim.Adam(disc_p.parameters(), lr=0.0002)

    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    # Включите режим обучения, если это необходимо
    if st.button('Начать обучение'):
        train(gen_ptm, gen_ptp, disc_m, disc_p, loader, opt_gen_ptm, opt_gen_ptp, opt_disc_m, opt_disc_p, l1_loss,
              mse_loss)

    # Перевод моделей в режим оценки
    gen_ptm.eval()
    gen_ptp.eval()
    disc_m.eval()
    disc_p.eval()

    # Streamlit интерфейс
    st.title('Генератор изображений в стиле Моне и Пикассо')
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_container_width=True)
        if st.button('Сгенерировать изображение'):
            # Преобразуйте изображение в массив NumPy
            image_np = np.array(image)

            # Примените трансформации
            image_tensor = transforms(image=image_np)['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                generated_image_ptm = gen_ptm(image_tensor)  # Генерация изображения в стиле Моне
                generated_image_ptp = gen_ptp(image_tensor)  # Генерация изображения в стиле Пикассо

            # Преобразуйте результат обратно в изображение
            generated_image_ptm = generated_image_ptm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_image_ptm = (generated_image_ptm * 255).astype('uint8')

            generated_image_ptp = generated_image_ptp.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_image_ptp = (generated_image_ptp * 255).astype('uint8')

            # Применение пастельного эффекта
            pastel_image_ptm = (generated_image_ptm * 0.8).clip(0, 255).astype('uint8')
            pastel_image_ptp = (generated_image_ptp * 0.8).clip(0, 255).astype('uint8')

            pastel_image_ptm = Image.fromarray(pastel_image_ptm)
            pastel_image_ptp = Image.fromarray(pastel_image_ptp)

            st.image(pastel_image_ptm, caption='Сгенерированное изображение в стиле Моне', use_container_width=True)
            st.image(pastel_image_ptp, caption='Сгенерированное изображение в стиле Пикассо', use_container_width=True)


if __name__ == '__main__':
    main()
