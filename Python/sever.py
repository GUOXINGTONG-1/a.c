from socket import *


def sever(code):
    HOST = '192.168.4.16'  # 服务器的ip，和上面的server.py对应
    PORT = 26206  # 相同的端口
    BUFSIZ = 1024
    ADDR = (HOST, PORT)

    tcpCliSock = socket(AF_INET, SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    stock_codes = code

    print("我是客户端发送的数据-->" + stock_codes)
    tcpCliSock.send(stock_codes.encode())  # 向服务器端发送编码后的数据
    stock_codes = tcpCliSock.recv(BUFSIZ).decode()  # 收到服务器端传过来的数据，解码
    print("我是服务器传回的数据-->" + stock_codes)
    tcpCliSock.close()
    return stock_codes
    '''
    while True:
        #stock_codes = input('the stock_codes is > ')  # 客户输入想输入的stock_codes
        stock_codes = code
        if not stock_codes:
            break
        tcpCliSock.send(stock_codes.encode())  # 向服务器端发送编码后的数据
        stock_codes = tcpCliSock.recv(BUFSIZ).decode()  # 收到服务器端传过来的数据，解码
        if not stock_codes:
            break
        print(stock_codes)
    
        after_close_simulation = input('the after_close_simulation is > ')
        if not after_close_simulation:
            break
        tcpCliSock.send(after_close_simulation.encode())  # 发送编码后的数据
        after_close_simulation = tcpCliSock.recv(BUFSIZ).decode()  # 收到数据，解码
        if not after_close_simulation:
            break
        print(stock_codes)
    '''
    tcpCliSock.close()