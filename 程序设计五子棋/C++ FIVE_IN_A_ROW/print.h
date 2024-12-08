// print.h头文件，是游戏主要过程的分步拆解
#ifndef PRINT_H
#define PRINT_H
#include "chess.h"
// 分解落子索引
void print_gochess(chessboard &board, int &x, int &y);

// 分解下棋
bool print_judge(int &num, chessboard &board, int &x, int &y);

// 分解打印棋盘格
void print_display(chessboard &board);

// 分解悔棋或者退出
bool print_retract(chessboard &board, int &num);

// 分解给出下棋建议
void print_tip(chessboard &board, const int &num);

// 分出胜负后，用户给出下一步操作
char print_final(chessboard &board, bool result, const int &num);

#endif