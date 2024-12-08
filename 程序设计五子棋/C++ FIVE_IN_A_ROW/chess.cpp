#include "chess.h"
#include <iomanip>
#include <windows.h>

// 构造大小为size的棋盘
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

// 析构函数，释放内存
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

// 打印棋盘格，为不同玩家设置了棋子颜色
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

// 判断(x,y)处落子合法性
bool chessboard::avail(int x, int y)
{
    if (x < 0 || x > 14 || y < 0 || y > 14)
    {
        cout << "请重新输入0~14?之间的正整数.";
        return false;
    }

    if (board[x][y].get_signal() == 'o') // 出现异常
        return true;

    else
        cout << "该处已有棋子,请重新选择落子位置.";
    return false;
}

// 在(x,y)处下棋
void chessboard::go_chess(int x, int y, char id)
{ // ????
    board[x][y].set_signal(id);
    record.push_back(board[x][y]);
}

// 悔棋，每次悔棋自动退回两步，即回到当前玩家所在的上一回合
void chessboard::retract()
{
    vector<chessnode>::iterator tmp = record.end() - 1;
    board[tmp->get_x()][tmp->get_y()].set_signal('o');
    record.pop_back();
    tmp = record.end() - 1;
    board[tmp->get_x()][tmp->get_y()].set_signal('o');
    record.pop_back();
}

// 保存棋盘
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

// 读取记录，并输入到当前棋盘中
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
     * @brief  判断连子长度
     * @param  x 棋子横坐标
     * @param  y 棋子纵坐标
     * @param  id 即假设该位置棋子为id
     * @param  sum 分解每个方向上棋子连成的长度
     * @return  max 该点所能连成的最大长度
     */

    int sum = 1, i = 1, j = 1, max = 0; // 自身点不用算，故都设置为1

    // 纵向
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

    // 横向
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

    // 主对角线
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

    // 副对角线
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

// 判断是否连成五子
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
     * @brief  评估玩家id合适的下棋位置，遍历棋盘上所有为空的位置，计算假设落子后的连子长度，由大到小排序后输出(tip)
     * @param  tmp 记录双方假设下棋后的连子长度
     * @param  value1 储存value1>=3的点
     * @param  value2 储存value2>=3的点
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
    // 冒泡排序value1
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
    // 冒泡排序value2
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
            cout << "如果您想进行攻击,可以选择以下位置:" << endl;
            for (auto p = value1.begin(); p != value1.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "没有建议的进攻位置." << endl;

        if (value2.size() != 0)
        {
            cout << "如果您想进行防守,可以选择以下位置:" << endl;
            for (auto p = value2.begin(); p != value2.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "没有建议的防守位置." << endl;
    }

    // player2
    else if (id == '#')
    {
        if (value2.size() != 0)
        {
            cout << "如果您想进行攻击,可以选择以下位置:" << endl;
            for (auto p = value2.begin(); p != value2.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "没有建议的进攻位置." << endl;

        if (value1.size() != 0)
        {
            cout << "如果您想进行防守,可以选择以下位置:" << endl;
            for (auto p = value1.begin(); p != value1.end(); ++p)
                cout << '(' << p->get_x() << ',' << p->get_y() << ')' << endl;
        }
        else
            cout << "没有建议的防守位置." << endl;
    }

    // 清空value1,value2,留下次计算使用
    value1.clear();
    value2.clear();
}
