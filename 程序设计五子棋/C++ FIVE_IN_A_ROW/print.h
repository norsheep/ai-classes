// print.hͷ�ļ�������Ϸ��Ҫ���̵ķֲ����
#ifndef PRINT_H
#define PRINT_H
#include "chess.h"
// �ֽ���������
void print_gochess(chessboard &board, int &x, int &y);

// �ֽ�����
bool print_judge(int &num, chessboard &board, int &x, int &y);

// �ֽ��ӡ���̸�
void print_display(chessboard &board);

// �ֽ��������˳�
bool print_retract(chessboard &board, int &num);

// �ֽ�������彨��
void print_tip(chessboard &board, const int &num);

// �ֳ�ʤ�����û�������һ������
char print_final(chessboard &board, bool result, const int &num);

#endif