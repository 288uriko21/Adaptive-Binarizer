### Первая, не особо удавшаяся попытка решения задачи
Идея в том, чтобы пытаться обрезать рамочку у картинок, где она есть, чтобы картинка стала более однородной в плане перепадов контрастности. После удаления рамочки, на основе предполагаемой зависимости delta от med (из функции _adaptive_median_threshold файла adaptive_binarizer.py) вычисляем delta : 

```
delta = round(((14400)/7) / ((255-medIm) + (20/7))) 
``` 
Размер блока = размеру картинки.



