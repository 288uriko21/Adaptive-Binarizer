### Вторая попытка решения задачи
Идея в том, чтобы пытаться обрезать рамочку у картинок, где она есть, чтобы картинка стала более однородной в плане перепадов контрастности. - Это не изменилось по сравнению с первым решением.

Изменилось то, что теперь программа нацелена на то, чтобы выбрать размер блока так, чтобы внутри блока по возможности не было перепадов оттенков фона. Также, пожалуй, стоит как-то более по-умному аппроксимировать f(GlobalMed) = delta. Эксперименты показали, что при выборе подходящего блока delta уменьшается с уменьшением контрастности текста. 




