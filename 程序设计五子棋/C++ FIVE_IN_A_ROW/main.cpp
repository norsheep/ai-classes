#include "print.h"
// �����⺯���������ռ����print.hͷ�ļ����Ѷ���

bool play(chessboard &board, int &num, char &ch) // ִ����Ϸ����
{
    int x, y;
    bool result;

    do
    {
        if (num % 2 == 0)
            cout << endl
                 << "���1ִ��." << endl;
        else
            cout << endl
                 << "���2ִ��." << endl;
        print_display(board);               // �Ƿ��ӡ���̵ĺ���
        result = print_retract(board, num); // �Ƿ��������˳���ֵĺ���
        if (!result)
            return false;           // ֻ����;����q���˳����
        print_tip(board, num);      // �Ƿ��ȡ����ĺ���
        print_gochess(board, x, y); // ȷ�����������ĺ���
        result = print_judge(num, board, x, y);
        // ���岢�жϸþ��Ƿ�����ĺ���������ֵΪ�ò�����������false�����������һ��225���������������Ǹò���δ�ֳ�ʤ��
    } while (!result && num < 225); // ѭ����result�Ǵ�Ĳ���num<225ʱ����ִ��

    ch = print_final(board, result, num);
    return true; // ������������������ֻ�᷵��true������;�˳���false���������������Ϸ�����Ӧ
}

int main()
{
    // һЩ��ʼ������
    const int n = 15;    // ��ʾ���̴�С
    chessboard board(n); // ��ʼ������
    int num = 0;         // numʵʱ��¼�����ϵ����Ӹ���
    char ch;             // ��������ÿ�����������
    bool result;         // ����һЩ��Ҫ�ж϶Դ�����
    ofstream outfile;    // ���ڱ����¼
    ifstream infile;     // ���ڶ�ȡ��¼

    // �Զ������Ϸ˵��
    cout << '\t' << "---����������---" << endl
         << endl;
    cout << "��Ϸ˵��" << endl;
    cout << "\t����һ��������������Ϸ\n"
            "��ÿһ�غ���,����0~14֮������������Ա�ʾ��������\n"
            "����r���Իص���ǰ������ڵ���һ�غ�(ע��: ÿ���ؿ�Ӧ�ú�,��Ҫ��������������������)\n"
            "����t���Ի�õ�ǰ����µĽ�������λ��(����Ϊ�ο�)\n"
            "����q�����˳���ֲ�����\n"
            "������ʤ���ѷ�,������ѡ�����ص���һ��,��Ҳ����ѡ�񱣴沢�˳�����,���ǲ���֧�ֶ�ȡ��¼�����(���ò���֧��)\n"
         << endl;

    // �Ƿ��ȡ�ϴμ�¼
    infile.open("save_file");
    if (!infile)
    {
        cout << "����s��ʼ��Ϸ" << endl;
        cin >> ch;
        while (ch != 's')
        {
            cout << endl
                 << "����ȷ����ָ��." << endl;
            cin >> ch;
        }
    }
    else
    {
        cout << "����s��ʼ����Ϸ,����i��ȡ��һ����Ϸ�浵��" << endl;
        cin >> ch;
        while (ch != 's' && ch != 'i')
        {
            cout << endl
                 << "����ȷ����ָ��." << endl;
            cin >> ch;
        }
        if (ch == 's')
            infile.close();
        else
            board.in(infile, n, num);
    }

    cout << "-------------------------------------------------------" << endl;

    // ��ʼ������Ϸ

AGAIN:                             // goto��־������ֳ�ʤ������岢������Ϸ���ص��ò�
    result = play(board, num, ch); // ��Ϸ��������Ϸ���������q���˳�����������false �����������true

    if (!result) // ��;����q�˳���ʱ��
    {
        outfile.open("save_file");   // ���ļ�
        board.save(outfile, n, num); // �����¼�ĺ���
        cout << "�þּ�¼�ѱ���." << endl;
        cout << "����س���������..." << endl;
        cin.get(ch);
        cin.get(ch);
        return 0; // �������
    }

    // result��true��������������ʤ���ѷ�
    switch (ch)
    {
    case 'r':
        board.retract(); // ����
        num = num - 2;
        goto AGAIN; // ������Ϸ
    case 'q':
        outfile.open("save_file");   // ���ļ�
        board.save(outfile, n, num); // �����¼�ĺ���
        cout << "�þּ�¼�ѱ���." << endl;
        cout << "����س���������..." << endl;
        cin.get(ch);
        cin.get(ch);
        return 0; // �������
    default:
        cout << "����س���������..." << endl;
        cin.get(ch);
        cin.get(ch);
    }

    return 0;
}
