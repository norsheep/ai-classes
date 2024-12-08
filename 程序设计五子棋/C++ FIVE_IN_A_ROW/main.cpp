#include "print.h"
// 其他库函数、命名空间等在print.h头文件中已定义

bool play(chessboard &board, int &num, char &ch) // 执行游戏函数
{
    int x, y;
    bool result;

    do
    {
        if (num % 2 == 0)
            cout << endl
                 << "玩家1执棋." << endl;
        else
            cout << endl
                 << "玩家2执棋." << endl;
        print_display(board);               // 是否打印棋盘的函数
        result = print_retract(board, num); // 是否悔棋或者退出棋局的函数
        if (!result)
            return false;           // 只有中途输入q会退出棋局
        print_tip(board, num);      // 是否获取建议的函数
        print_gochess(board, x, y); // 确定落子索引的函数
        result = print_judge(num, board, x, y);
        // 下棋并判断该局是否结束的函数，返回值为该步后比赛结果：false有两种情况：一是225个格子下满，二是该步后未分出胜负
    } while (!result && num < 225); // 循环，result是错的并且num<225时候能执行

    ch = print_final(board, result, num);
    return true; // 对于整个函数来讲，只会返回true，与中途退出的false（玩家主动结束游戏）相对应
}

int main()
{
    // 一些初始化内容
    const int n = 15;    // 表示棋盘大小
    chessboard board(n); // 初始化棋盘
    int num = 0;         // num实时记录棋盘上的棋子个数
    char ch;             // 用来代表每次输入的命令
    bool result;         // 用于一些需要判断对错的情况
    ofstream outfile;    // 用于保存记录
    ifstream infile;     // 用于读取记录

    // 自动输出游戏说明
    cout << '\t' << "---欢乐五子棋---" << endl
         << endl;
    cout << "游戏说明" << endl;
    cout << "\t这是一个简易五子棋游戏\n"
            "在每一回合中,输入0~14之间的正整数数对表示落子索引\n"
            "输入r可以回到当前玩家所在的上一回合(注意: 每次重开应用后,需要至少下两步棋才允许悔棋)\n"
            "输入t可以获得当前棋局下的建议落子位置(仅作为参考)\n"
            "输入q可以退出棋局并保存\n"
            "如果棋局胜负已分,您可以选择悔棋回到上一步,您也可以选择保存并退出程序,但是并不支持读取记录后悔棋(仅该步不支持)\n"
         << endl;

    // 是否读取上次记录
    infile.open("save_file");
    if (!infile)
    {
        cout << "输入s开始游戏" << endl;
        cin >> ch;
        while (ch != 's')
        {
            cout << endl
                 << "请正确输入指令." << endl;
            cin >> ch;
        }
    }
    else
    {
        cout << "输入s开始新游戏,输入i读取上一次游戏存档。" << endl;
        cin >> ch;
        while (ch != 's' && ch != 'i')
        {
            cout << endl
                 << "请正确输入指令." << endl;
            cin >> ch;
        }
        if (ch == 's')
            infile.close();
        else
            board.in(infile, n, num);
    }

    cout << "-------------------------------------------------------" << endl;

    // 开始进行游戏

AGAIN:                             // goto标志，如果分出胜负后悔棋并继续游戏，回到该步
    result = play(board, num, ch); // 游戏函数，游戏中玩家输入q会退出，函数返回false 其他情况返回true

    if (!result) // 中途输入q退出的时候
    {
        outfile.open("save_file");   // 打开文件
        board.save(outfile, n, num); // 保存记录的函数
        cout << "该局记录已保存." << endl;
        cout << "输入回车结束程序..." << endl;
        cin.get(ch);
        cin.get(ch);
        return 0; // 程序结束
    }

    // result是true，棋盘下满或者胜负已分
    switch (ch)
    {
    case 'r':
        board.retract(); // 悔棋
        num = num - 2;
        goto AGAIN; // 继续游戏
    case 'q':
        outfile.open("save_file");   // 打开文件
        board.save(outfile, n, num); // 保存记录的函数
        cout << "该局记录已保存." << endl;
        cout << "输入回车结束程序..." << endl;
        cin.get(ch);
        cin.get(ch);
        return 0; // 程序结束
    default:
        cout << "输入回车结束程序..." << endl;
        cin.get(ch);
        cin.get(ch);
    }

    return 0;
}
