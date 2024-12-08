// chessnode和chessboard的头文件
#ifndef CHESS_H
#define CHESS_H
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
class chessnode // 棋子类
{
private:
    int location_x; // 横坐标
    int location_y; // 纵坐标
    char signal;    //  玩家一棋子%    玩家二棋子#
    int value1, value2;
    //  评估值，即在该点落子后能连成的长度
    //  value1 %  value2 #

public:
    chessnode() // 构造函数
    {
        signal = 'o';
        value1 = 0;
        value2 = 0;
    }
    void set_chessnode(int x, int y) // 设置横纵坐标
    {
        location_x = x;
        location_y = y;
    }
    int get_x() const // 获取横坐标
    {
        return location_x;
    }
    int get_y() const // 获取纵坐标
    {
        return location_y;
    }
    void set_signal(char ch) // 设置棋子状况
    {
        signal = ch;
    }
    char get_signal() const // 返回棋子状况
    {
        return signal;
    }
    void set_value1(int x) // 设置value1
    {
        value1 = x;
    }
    int get_value1() const // 返回value1
    {
        return value1;
    }
    void set_value2(int x) // 设置value2
    {
        value2 = x;
    }
    int get_value2() const // 返回value2
    {
        return value2;
    }
};

class chessboard // 棋盘类
{
private:
    chessnode **board;                // 二维指针，指向棋盘
    int size;                         // 棋盘大小
    vector<chessnode> record;         // 记录下棋顺序
    vector<chessnode> value1;         // 记录player1能连成长度大于等于3的棋子
    vector<chessnode> value2;         // 记录player2能连成长度大于等于3的棋子
    int value(int x, int y, char id); // 判断棋盘(x,y)处落子id后能连成的最大长度

public:
    chessboard(int size); // 构造函数

    ~chessboard(); // 析构函数

    void display(); // 打印棋盘格

    bool avail(int x, int y); // 判断落子合法性

    void go_chess(int x, int y, char id); // 下棋

    bool judge(int x, int y, char id); // 判断是否连成五子

    void save(ofstream &outfile, const int &n, const int &num); // 保存棋盘

    void in(ifstream &infile, const int &n, int &num); // 读取棋盘

    void retract(); // 悔棋

    void evaluate(char id); // 评估玩家id合适的下棋位置

    int record_num() // 返回当前储存的下棋顺序个数
    {
        return record.size();
    }
};

#endif
// CHESS_H
