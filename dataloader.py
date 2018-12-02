import sqlite3 as sql
from collections import namedtuple


LotteryRecord = namedtuple('LotteryRecord', ['date','hundred','decade', 'unit', 'sales'])


class LotteryRecordLoad(object):
    def __init__(self, db):
        super(LotteryRecordLoad, self).__init__()
        self.conn = sql.connect(db)
        self.cur = self.conn.cursor()

    def fetch_items_from_table(self, table, type_item, order_by=None, reverse=False, num=-1):
        columns = [key for key in type_item._fields]
        placeholder = ','.join(columns)
        if order_by:
            if reverse:
                order_by = 'order by {} desc'.format(order_by)
            else:
                order_by = 'order by {}'.format(order_by)
        else:
            order_by = ''

        if num > 0:
            limit_num = 'limit {}'.format(num)
        else:
            limit_num = ''

        command_line = 'select {} from {} {} {}'.format(placeholder, table, order_by, limit_num)
        cursor = self.cur.execute(command_line)
        rtn_items = [type_item._make(ele) for ele in cursor]
        return rtn_items

    def __call__(self, table, type_item, order_by=None, reverse=False,num=-1):
        return self.fetch_items_from_table(table, type_item, order_by, reverse, num)

    def __del__(self):
        self.conn.close()


if __name__ == '__main__':
    items = LotteryRecordLoad(r'C:\Users\gni\Desktop\tutorial\LotteryRecode.sqlite')
    ele_list = items('lottery', LotteryRecord, order_by='date', reverse=True, num=5)
    print(ele_list)


