#include "print.h"

void print_gochess(chessboard &board, int &x, int &y)
{
    /**
     * @brief  �û����������������ж��Ƿ�Ϸ����Ƿ�ȷ������
     * @param  x ���Ӻ�����
     * @param  y ����������
     * @param  ch �û���������
     * @param  result �ж����ӺϷ���
     */
    char ch;
    bool result;
    do
    {
        do
        {
            cout << "��������������(��һ������Ϊ��������,�ڶ�����Ϊ��������)" << endl;
            cin >> x >> y;
            result = board.avail(x, y); // �жϺϷ���

        } while (!result);
        do
        {
            cout << "�Ƿ�ȷ������?ȷ��������y,��֮������n." << endl;
            cin >> ch;
        } while (ch != 'y' && ch != 'n');

    } while (ch != 'y');
}

bool print_judge(int &num, chessboard &board, int &x, int &y)
{
    /**
     * @brief  �ж�����Ƿ����
     * @param  x ���Ӻ�����
     * @param  y ����������
     * @param  num ���������Ӹ���
     * @param  result �ж�����Ƿ����
     * @return true��ʾ����һ�ʤ��false��ʾʤ��δ�ֻ�����������
     */
    bool result;
    if (num % 2 == 0)
    {
        board.go_chess(x, y, '%');       // ���һ
        result = board.judge(x, y, '%'); // �ж����
    }
    if (num % 2 == 1)
    {
        board.go_chess(x, y, '#'); // ��Ҷ�
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
     * @brief  �û�ѡ���Ƿ��ӡ���̸�
     * @param  ch �û���������
     */
    char ch;
    do
    {
        cout << "�����ӡ���̸�,������p,��֮������n." << endl;
        cin >> ch;
        if (ch == 'p')
            board.display(); // ��ӡ���̸�
    } while (ch != 'p' && ch != 'n');
}

bool print_retract(chessboard &board, int &num)
{
    /**
     * @brief  �û�ѡ���Ƿ������˳����
     * @param  num ���������Ӹ���
     * @return false��ʾ�˳����
     */
    char ch;

    do
    {
        cout << "����Ҫ����,������r,���˳���Ϸ,������q,������,������n."
             << endl;
        cin >> ch;
        if (ch == 'r')
        {
            if (board.record_num() <= 2)
                cout << "����ʧ��." << endl;
            else
            {
                board.retract();
                num = num - 2;
                cout << "�����,����������ʾ." << endl;
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
     * @brief  �û�ѡ���Ƿ���Ҫ���ӽ���
     * @param  ch �û���������
     * @param  num ���������Ӹ���
     */
    char ch;

    do
    {
        cout << "�Ƿ���Ҫ����,����Ҫ,������t,��֮,������n."
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

// �ֳ�ʤ�����û�������һ������
char print_final(chessboard &board, bool result, const int &num)
{

    char ch;
    board.display();
    if (result)
    {
        if (num % 2 == 1)
            cout << "���1��ʤ!" << endl;
        else
            cout << "���2��ʤ!" << endl;
    }
    else
        cout << "��������,ʤ��δ��,�þ�ƽ��." << endl;

    cout << endl
         << "����������r,����q�����¼���˳���Ϸ,���������˳���Ϸ." << endl;
    cin >> ch;
    return ch;
}
