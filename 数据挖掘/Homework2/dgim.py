import os

# path to data file
file_dir = os.path.dirname(__file__)
data_path = os.path.join(file_dir, "stream_data_dgim.txt")
edge_path1 = os.path.join(file_dir, "edge1.txt")
edge_path2 = os.path.join(file_dir, "edge2.txt")


class DGIM:

    def __init__(self, filepath, windowsize, maxtime=None):
        '''init DGIM for specific file

        Args:
            filepath (str): data file path
            windowsize (int): 
            maxtime (int, optional): timestamp modulo N. Defaults to None.
        '''
        self.fileHandler = open(filepath, 'r')
        self.windowSize = windowsize
        self.buckets = []  # list[list]
        self.timeMod = maxtime if maxtime else windowsize << 2
        self.timestamp = 0

    def update(self, x):
        '''update when a new bit come in
        Args:
            x (str): new bit, can be "1" or "0"
        '''
        ### TODO
        ### maintaining 1 or 2 of each size bucket
        # 如果超出时间窗，删掉
        if self.buckets and (self.timestamp - self.windowSize + self.timeMod
                             ) % self.timeMod == self.buckets[0][0]:
            # 相等的时候已经不在里面了，一个一个加一定会有相等
            del self.buckets[0]

        if x == '0':
            return

        # else: x=1
        self.buckets.append([self.timestamp, 1])
        # remove old buckets
        for i in range(len(self.buckets) - 1, 1, -1):
            # 合并相同大小的桶。三个时合并，桶大小递减，只需要看前两个，三个一样就把前两个合并
            if self.buckets[i - 2][1] == self.buckets[i][1]:
                # 合并
                # print('before', self.buckets)
                self.buckets[i - 1][1] <<= 1
                del self.buckets[i - 2]
                i -= 1
                # print('after',self.buckets)

        ### end of TODO

    def run(self):
        '''simulate the process of stream data
        '''
        f = self.fileHandler
        x = f.read(2).strip()
        while x:
            # x can be string "1" or "0"
            self.update(x)

            # get next bit
            self.timestamp = (self.timestamp + 1) % self.timeMod
            x = f.read(2).strip()

    def count(self, k=None):
        '''count the number of 1-bits in last k bits

        Args:
            k (int, optional): . Defaults to the windowsize.

        Returns:
            int: count results
        '''

        if k is None:
            k = self.windowSize

        result = 0

        ### TODO
        ### return floor(1 / 2) if the last bucket is zero
        # 检查前k个bit里有多少个1，可能会小于或者大于滑动窗口。所以要看当前在不在k里面
        # 先全都加上
        idx = len(self.buckets) - 1
        if idx != -1:
            while idx >= 0 and (self.timestamp - k + self.timeMod
                                ) % self.timeMod < self.buckets[idx][0]:
                result += self.buckets[idx][1]
                idx -= 1
            # 没法判断最后一个桶是不是完全在k里面，因为不是真实的size，所以必须得减去一半
            result -= self.buckets[idx + 1][1] // 2  # 如果len=0 bug

        return result
        ### end of TODO


if __name__ == "__main__":

    dgim = DGIM(filepath=data_path, windowsize=1000)

    dgim.run()
    #print(dgim.timestamp, dgim.buckets)

    # current window
    print("the number of 1-bits in current windows: ")
    print(f"   {dgim.count()}")

    # last 500 and 200
    print("the number of 1-bits in the last 500 and 200 bits of the stream")
    print(f"   {dgim.count(k=500)}")
    print(f"   {dgim.count(k=200)}")

    print("edge cases:")
    dgim1 = DGIM(filepath=edge_path1, windowsize=1000)

    dgim1.run()

    #print(dgim1.timestamp, dgim1.buckets)
    print(f"   {dgim1.count()}")
    dgim2 = DGIM(filepath=edge_path2, windowsize=1000)

    dgim2.run()
    #print(dgim2.timestamp, dgim2.buckets)
    print(f"   {dgim2.count()}")
