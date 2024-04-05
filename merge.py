while True:
    f1 = open('fer_txt.txt', 'r')
    c1 = f1.read()
    f2 = open('ser_txt.txt', 'r')
    c2 = f2.read()
    file = open('final_txt.txt', 'w')

    print(c1)
    print(c2)

    TD = 0 #TenDency
    if c1 == '':
        c1 = '5'
    if int(c1) == 4:
        TD = 0 #positive
        if c2 == 'Angry' or c2 == 'Disgust' or c2 == 'Fear' or c2 == 'Sad':
            file.write('0')
        else:
            file.write(c1)
    elif int(c1) == 5 or int(c1) == 7:
        TD = 1 #neutral
        file.write(c1)
    else:
        TD = 2 #negative
        if c2 == 'Happy':
            file.write('0')
        else:
            file.write(c1)
