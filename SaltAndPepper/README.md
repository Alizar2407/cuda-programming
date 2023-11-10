# Salt and Pepper filtering
---
Было произведено сравнение скорости работы итеративного (с использованием CPU) и параллельного (с использованием GPU) алгоритмов на изображениях с разным разрешением, от 64x96 до 1024x1536. Время работы обоих алгоритмов указано над отфильтрованными изображениями.

---
## Изображение размером 64x96

![sap_filtering_64](/assets/sap_filtering_64.png)

---
## Изображение размером 128x192

![sap_filtering_128](/assets/sap_filtering_128.png)

---
## Изображение размером 256x384


![sap_filtering_256](/assets/sap_filtering_256.png)

---
## Изображение размером 512x758

![sap_filtering_512](/assets/sap_filtering_512.png)

---
## Изображение размером 1024x1536

![sap_filtering_1024](/assets/sap_filtering_1024.png)

---
## График сравнения скоростей алгоритмов

![cpu_gpu_comparison](/assets/sap_filtering_cpu_gpu_comparison.png)

---
## Вывод

Для итеративного алгоритма время обработки изображения прямо пропорционально его размеру (при увеличении размера с *w x h* до *2w x 2h* время обработки возрастает в 4 раза).

В то же время, для алгоритма, использующего GPU, время обработки  изображения почти не зависит от его размера (до тех пор, пока достаточно памяти для размещения нужного количества блоков).