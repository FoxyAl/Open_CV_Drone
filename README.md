# OpenCV_filter

  Это приложение демонстрирует обработку видео в реальном времени с добавлением шума соль/перец и последующей фильтрацией медианной сглаживаеем. 
При запуске программа показывает три варианта изображения: оригинальное, зашумленное и отфильтрованное.

  Полученное изображение мы сначально заполняем на 20% шумом соль/перец (черные и белые пиксели). Далее проходимся медианным фильтром, удаляя шум соль/перец добавленный ранее матрицей 5х5 пикселей. После чего используем временное сглаживание (temporal smoothing), которое сохраняет предыдуший кадр, смешивается с новым кадром по формуле:
filtered_frame = alpha * current_frame + (1 - alpha) * previous_frame
и результат сохраняется для следующего кадра.

Для выхода из программы необходимо использовать английскую раскладку клавиатуры и нажать кнопку "E".

Требования:
1. Python 3.8
2. Numpy
3. OpenCV
