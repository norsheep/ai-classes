#include "chess.h"
#include <iomanip>
#include <windows.h>

// �����СΪsize������
chessboard::chessboard(int size)
{
    this->size = size;
    board = new chessnode *[size];
    for (int i = 0; i < size; ++i)
    {
        board[i] = new chessnode[size];
        for (int j = 0; j < size; ++j)
        {
            board[i][j].set_chessnode(i, j); // %player1  #player2
        }
    }
}

// �����������ͷ��ڴ�
chessboard::~chessboard()
{
    for (int i = 0; i < size; ++i)
    {
        delete[] board[i];
        board[i] = nullptr;
    }
    delete[] board;
    board = nullptr;
}

// ��ӡ���̸�Ϊ��ͬ���������������ɫ
void chessboard::display()
{
    cout << "   ";
    for (int i = 0; i < size; ++i)
        cout << setw(3) << left << i;
    cout << endl;
    for (int i = 0; i < size; ++i)
    {

        cout << setw(3) << left << i;
        for (int j = 0; j < size; ++j)
        {
            if (board[i][j].get_signal() == '%')
            {
                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_BLUE | FOREGROUND_GREEN);
                cout << setw(3) << left << board[i][j].get_signal();
                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE);
            }
            else if (board[i][j].get_signal() == '#')
            {
                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED);
                cout << setw(3) << left << board[i][j].get_signal();
                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE);
            }
            else if (board[i][j].get_signal() == 'o')
                cout << setw(3) << left << board[i][j].get_signal();
        }
        cout << endl;
    }
}

// �ж�(x,y)�����ӺϷ���
bool chessboard::avail(int x, int y)
{
    if (x < 0 || x > 14 || y < 0 || y > 14)
    {
        cout << "����������0~14?֮���������.";
        return false;
    }

    if (board[x][y].get_signal() == 'o') // �����쳣
        return true;

    else
        cout << "�ô���������,������ѡ������λ��.";
    return false;
}

// ��(x,y)������
void chessboard::go_chess(int x, int y, char id)
{ // ????
    board[x][y].set_signal(id);
    record.push_back(board[x][y]);
}

// ���壬ÿ�λ����Զ��˻����������ص���ǰ������ڵ���һ�غ�
void chessboard::retract()
{
    vector<chessnode>::iterator tmp = record.end() - 1;
    board[tmp->get_x()][tmp->get_y()].set_signal('o');
    record.pop_back();
    tmp = record.end() - 1;
    board[tmp->get_x()][tmp->get_y()].set_signal('o');
    record.pop_back();
}

// ��������
void chessboard::save(ofstream &outfile, const int &n, const int &num)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            outfile << board[i][j].get_signal() << endl;
        }
    }
    outfile << num << endl;
    outfile.close();
}

// ��ȡ��¼�������뵽��ǰ������
void chessboard::in(ifstream &infile, const int &n, int &num)
{
    char ch;
    int x;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            infile >> ch;
            board[i][j].set_signal(ch);
        }
    }

    if (infile >> x)
        num = x;
    infile.close();
}

int chessboard::value(int x, int y, char id)
{
    /**
     * @brief  �ж����ӳ���
     * @param  x ���Ӻ�����
     * @param  y ����������
     * @param  id �������λ������Ϊid
     * @param  sum �ֽ�ÿ���������������ɵĳ���
     * @return  max �õ��������ɵ���󳤶�
     */

    int sum = 1, i = 1, j = 1, max = 0; // ����㲻���㣬�ʶ�����Ϊ1

    // ����
    while (x - i >= 0)
    {
        if (board[x - i][y].get_signal() == id)
            ++sum;
        else
            break;
        ++i;
    }
    while (x + j <= 14)
    {
        if (board[x + j][y].get_signal() == id)
            ++sum;
        else
            break;
        ++j;
    }
    if (sum > max)
        max = sum;

    // ����
    sum = 1;
    i = 1;
    j = 1;
    while (y - i >= 0)
    {
        if (board[x][y - i].get_signal() == id)
            ++sum;
        else
            break;
        ++i;
    }
    while (y + j <= 14)
    {
        if (board[x][y + j].get_signal() == id)
            ++sum;
        else
            break;
        ++j;
    }
    if (sum > max)
        max = sum;

    // ���Խ���
    sum = 1;
    i = 1;
    j = 1;
    while (x - i >= 0 && y - i >= 0)
    {
        if (board[x - i][y - i].get_signal() == id)
            ++sum;
        else
            break;
        ++i;
    }
    while (x + j <= 14 && y + j <= 14)
    {
        if (board[x + j][y + j].get_signal() == id)
            ++sum;
        else
            break;
        ++j;
    }

    if (sum > max)
        max = sum;

    // ���Խ���
    sum = 1;
    i = 1;
    j = 1;
    while (x + i <= 14 && y - i >= 0)
    {
        if (board[x + i][y - i].get_signal() == id)
            ++sum;
        else
            break;
        ++i;
    }
    while (x - j >= 0 && y + j <= 14)
    {
        if (board[x - j][y + j].get_signal() == id)
            ++sum;
        else
            break;
        ++j;
    }
    if (sum > max)
        max = sum;
    return max;
}

// �ж��Ƿ���������
bool chessboard::judge(int x, int y, char id)
{
    int tmp = value(x, y, id);
    if (tmp >= 5)
        return true;
    return false;
}

void chessboard::evaluate(char id)
{
    /**
     * @brief  �������id���ʵ�����λ�ã���������������Ϊ�յ�λ�ã�����������Ӻ�����ӳ��ȣ��ɴ�С��������(tip)
     * @param  tmp ��¼˫���������������ӳ���
     * @param  value1 ����value1>=3�ĵ�
     * @param  value2 ����value2>=3�ĵ�
     */

    int tmp, i;
    for (int x = 0; x < 15; ++x)
    {
        for (int y = 0; y < 15; ++y)
        {
            if (board[x][y].get_signal() == 'o')
            {
                // player1
                tmp = value(x, y, '%');
                if (tmp >= 3)
                {
                    board[x][y].set_value1(tmp);
                    value1.push_back(board[x][y]);
                }

                // player2
                tmp = value(x, y, '#');
                if (tmp >= 3)
                {
                    board[x][y].set_value2(tmp);
                    value2.push_back(board[x][y]);
                }
            }
        }
    }
    // ð������value1
    if (value1.size() != 0)
    {
        for (i = 0; i < value1.size() - 1; ++i)
        {
            for (auto p = value1.begin(); p != value1.end() - 1 - i; ++p)
            {
                if (p->get_value1() < (p + 1)->get_value1())
                {
                    chessnode chess = *p;
                    *p = *(p + 1);
                    *(p + 1) = chess;
                }
            }
        }
    }
    // ð������value2
    if (value2.size() != 0)
    {
        for (i = 0; i < value2.size() - 1; ++i)
        {
            for (auto p = value2.begin(); p != value2.end() - 1 - i; ++p)
            {
                if (p->get_value2() < (p + 1)->get_value2())
                {
                    chessnode chess = *p;
                    *p = *(p + 1);
                    *(p + 1) = chess;
                }
            }
        }
    }

    // player1
    if (id == '%')
    {
        if (value1.size() != 0)
        {
            cout << "���������й���,����ѡ������λ��:" << endl;
            for (auto p = value1.begin(); p != value1.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "û�н���Ľ���λ��." << endl;

        if (value2.size() != 0)
        {
            cout << "���������з���,����ѡ������λ��:" << endl;
            for (auto p = value2.begin(); p != value2.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "û�н���ķ���λ��." << endl;
    }

    // player2
    else if (id == '#')
    {
        if (value2.size() != 0)
        {
            cout << "���������й���,����ѡ������λ��:" << endl;
            for (auto p = value2.begin(); p != value2.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "û�н���Ľ���λ��." << endl;

        if (value1.size() != 0)
        {
            cout << "���������з���,����ѡ������λ��:" << endl;
            for (auto p = value1.begin(); p != value1.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "û�н���ķ���λ��." << endl;
    }

    // ���value1,value2,���´μ���ʹ��
    value1.clear();
    value2.clear();
}
