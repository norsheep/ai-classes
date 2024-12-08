// chessnode��chessboard��ͷ�ļ�
#ifndef CHESS_H
#define CHESS_H
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
class chessnode // ������
{
private:
    int location_x; // ������
    int location_y; // ������
    char signal;    //  ���һ����%    ��Ҷ�����#
    int value1, value2;
    //  ����ֵ�����ڸõ����Ӻ������ɵĳ���
    //  value1 %  value2 #

public:
    chessnode() // ���캯��
    {
        signal = 'o';
        value1 = 0;
        value2 = 0;
    }
    void set_chessnode(int x, int y) // ���ú�������
    {
        location_x = x;
        location_y = y;
    }
    int get_x() const // ��ȡ������
    {
        return location_x;
    }
    int get_y() const // ��ȡ������
    {
        return location_y;
    }
    void set_signal(char ch) // ��������״��
    {
        signal = ch;
    }
    char get_signal() const // ��������״��
    {
        return signal;
    }
    void set_value1(int x) // ����value1
    {
        value1 = x;
    }
    int get_value1() const // ����value1
    {
        return value1;
    }
    void set_value2(int x) // ����value2
    {
        value2 = x;
    }
    int get_value2() const // ����value2
    {
        return value2;
    }
};

class chessboard // ������
{
private:
    chessnode **board;                // ��άָ�룬ָ������
    int size;                         // ���̴�С
    vector<chessnode> record;         // ��¼����˳��
    vector<chessnode> value1;         // ��¼player1�����ɳ��ȴ��ڵ���3������
    vector<chessnode> value2;         // ��¼player2�����ɳ��ȴ��ڵ���3������
    int value(int x, int y, char id); // �ж�����(x,y)������id�������ɵ���󳤶�

public:
    chessboard(int size); // ���캯��

    ~chessboard(); // ��������

    void display(); // ��ӡ���̸�

    bool avail(int x, int y); // �ж����ӺϷ���

    void go_chess(int x, int y, char id); // ����

    bool judge(int x, int y, char id); // �ж��Ƿ���������

    void save(ofstream &outfile, const int &n, const int &num); // ��������

    void in(ifstream &infile, const int &n, int &num); // ��ȡ����

    void retract(); // ����

    void evaluate(char id); // �������id���ʵ�����λ��

    int record_num() // ���ص�ǰ���������˳�����
    {
        return record.size();
    }
};

#endif
// CHESS_H
