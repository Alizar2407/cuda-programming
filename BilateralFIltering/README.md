# Bilateral filtering
---
Было произведено сравнение скорости работы итеративного (с использованием CPU) и параллельного (с использованием GPU) алгоритмов на изображениях с разным разрешением, от 128x128 до 1024x1024. Размер ядра фильтра в каждом случае был равен 7, поэтому размытие более заметно для изображений с небольшим разрешением. Время работы обоих алгоритмов указано над размытыми изображениями.

---
## Изображение размером 128x128

![bilateral_filtering_128](/assets/bilateral_filtering_128.png)

---
## Изображение размером 256x256


![bilateral_filtering_256](/assets/bilateral_filtering_256.png)

---
## Изображение размером 512x512

![bilateral_filtering_512](/assets/bilateral_filtering_512.png)

---
## Изображение размером 1024x1024

![bilateral_filtering_1024](/assets/bilateral_filtering_1024.png)

---
## График сравнения скоростей алгоритмов

![cpu_gpu_comparison](/assets/bilateral_filtering_cpu_gpu_comparison.png)

---
## Вывод

Для итеративного алгоритма время обработки изображения прямо пропорционально его размеру (при увеличении размера с 128x128 до 256x256 время обработки возрастает в 4 раза).

В то же время, для алгоритма, использующего GPU, время обработки  изображения почти не зависит от его размера (до тех пор, пока достаточно памяти для размещения нужного количества блоков).
