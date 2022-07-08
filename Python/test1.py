# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:06:24 2022

@author: lijinchao-002
"""
import time
from wxRobot import WeChatRobot
from Python import sever

def ReceiveMessageCallBack(robot,message):
    print(message['wxid'])
    #
    if message['type'] == 1 and message['sender'] != 'filehelper':
        #robot.robot.CSendText('filehelper',message['message'])
        robot.robot.CSendText(message['sender'], message['message'])
    wxSender = robot.GetWxUserInfo(message['sender'])
    sender = wxSender['wxNickName'] if wxSender['wxNickName'] != 'null' else message['sender']
    if '@chatroom' in message['sender']:
        wxUser = robot.GetWxUserInfo(message['wxid'])
        print("来自 {} {},type {}".format(sender,wxUser['wxNickName'],message['type']))
    else:
        print("来自 {},type {}".format(sender,message['type']))
    if message['type'] == 1:
        print(message['message'])
    elif message['type'] == 3:
        print(message['message'])
        print(message['filepath'])
    elif message['type'] == 49:
        print(message['message'])
        if not message['filepath']: print(message['filepath'])
    else:
        print(message['message'])

def test_Message():
    wx = WeChatRobot()
    wx.StartService()
    wx.StartReceiveMessage(MessageCallBack)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    wx.StopService(wx)

def MessageCallBack(robot,message):
    #print(message['message'])
    #wxSender = robot.GetWxUserInfo(message['sender'])
    #print(wxSender)
    getword = message['message']
    answer = sever.sever(getword)
    #robot.robot.CSendText(message['sender'], answer + '\n测试中！！！！！！！')
    robot.robot.CSendText(message['sender'], answer)


'''
    if getword == '1':
        robot.robot.CSendText(message['sender'], '它是1')
    else:
        robot.robot.CSendText(message['sender'], '它不是1')
'''


if __name__ == '__main__':
    test_Message()
    
