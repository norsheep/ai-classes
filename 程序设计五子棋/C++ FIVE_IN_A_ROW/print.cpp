#include "print.h"

void print_gochess(chessboard &board, int &x, int &y)
{
    /**
     * @brief  用户输入落子索引，判断是否合法，是否确定输入
     * @param  x 落子横坐标
     * @param  y 落子纵坐标
     * @param  ch 用户输入命令
     * @param  result 判断落子合法性
     */
    char ch;
    bool result;
    do
    {
        do
        {
            cout << "请输入落子索引(第一个整数为横轴坐标,第二个数为纵轴坐标)" << endl;
            cin >> x >> y;
            result = board.avail(x, y); // 判断合法性

        } while (!result);
        do
        {
            cout << "是否确认落子?确认请输入y,反之请输入n." << endl;
            cin >> ch;
        } while (ch != 'y' && ch != 'n');

    } while (ch != 'y');
}

bool print_judge(int &num, chessboard &board, int &x, int &y)
{
    /**
     * @brief  判断棋局是否结束
     * @param  x 落子横坐标
     * @param  y 落子纵坐标
     * @param  num 棋盘上棋子个数
     * @param  result 判断棋局是否结束
     * @return true表示有玩家获胜，false表示胜负未分或者棋盘已满
     */
    bool result;
    if (num % 2 == 0)
    {
        board.go_chess(x, y, '%');       // 玩家一
        result = board.judge(x, y, '%'); // 判断棋局
    }
    if (num % 2 == 1)
    {
        board.go_chess(x, y, '#'); // 玩家二
        result = board.judge(x, y, '#');
    }

    ++num;
    if (num == 225)
        return false;
    return result;
}

void print_display(chessboard &board)
{
    /**
     * @brief  用户选择是否打印棋盘格
     * @param  ch 用户输入命令
     */
    char ch;
    do
    {
        cout << "若想打印棋盘格,请输入p,反之请输入n." << endl;
        cin >> ch;
        if (ch == 'p')
            board.display(); // 打印棋盘格
    } while (ch != 'p' && ch != 'n');
}

bool print_retract(chessboard &board, int &num)
{
    /**
     * @brief  用户选择是否悔棋或退出棋局
     * @param  num 棋盘上棋子个数
     * @return false表示退出棋局
     */
    char ch;

    do
    {
        cout << "若需要悔棋,请输入r,若退出游戏,请输入q,若均不,请输入n."
             << endl;
        cin >> ch;
        if (ch == 'r')
        {
            if (board.record_num() <= 2)
                cout << "悔棋失败." << endl;
            else
            {
                board.retract();
                num = num - 2;
                cout << "悔棋后,棋盘如下所示." << endl;
                board.display();
            }
        }
        if (ch == 'q')
            return false;
    } while (ch != 'r' && ch != 'n' && ch != 'q');

    return true;
}

void print_tip(chessboard &board, const int &num)
{
    /**
     * @brief  用户选择是否需要落子建议
     * @param  ch 用户输入命令
     * @param  num 棋盘上棋子个数
     */
    char ch;

    do
    {
        cout << "是否需要建议,若需要,请输入t,反之,请输入n."
             << endl;
        cin >> ch;
        if (ch == 't')
        {
            if (num % 2 == 0)
                board.evaluate('%'); //
            else
                board.evaluate('#');
        }

    } while (ch != 't' && ch != 'n');
}

// 分出胜负后，用户给出下一步操作
char print_final(chessboard &board, bool result, const int &num)
{

    char ch;
    board.display();
    if (result)
    {
        if (num % 2 == 1)
            cout << "玩家1获胜!" << endl;
        else
            cout << "玩家2获胜!" << endl;
    }
    else
        cout << "棋盘已满,胜负未分,该局平局." << endl;

    cout << endl
         << "悔棋请输入r,输入q保存记录并退出游戏,输入其他退出游戏." << endl;
    cin >> ch;
    return ch;
}
