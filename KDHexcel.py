import win32com.client
import io
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import pandas
import numpy

class KDHexcel:
    __version__ = 1.0
    __작성자__ = "김대호"
    __버전__ = __version__

    def _open(self):
        excel = win32com.client.Dispatch("Excel.Application")  # 엑셀 어플리케이션 선언 """
        excel.Visible = True  # 엑셀 프로그램 보여지게 함 """

        if excel.Workbooks.Count == 0:  # 워크북 새로 만듦 """ # 없으면 새로 만듦
            wb = excel.Workbooks.Add()
        else:  # 있으면 마지막 엑셀 문서
            wb = excel.Workbooks(excel.Workbooks.Count)
        return excel, wb

    def _close(self, ws, excel):
        """ 선언한 엑셀 app, 워크북 초기화 """
        ws = None
        excel = None

    def _시트명(self, wb, 시트명=None):
        """ 시트명이 입력 되면 해당 시트 선택 없으면 시트생성, 시트명이 입력 되지 않으면 첫번째 시트명 리턴 """

        if 시트명 is None:
            ws = wb.Worksheets(1)
            return ws
        else:
            try:
                ws = wb.Worksheets(시트명)
                return ws
            except:
                ws = wb.Sheets.Add(After:=wb.Sheets(wb.Sheets.Count))
                ws.Name = 시트명
                return ws


    def 셀값넣기(self, 셀="A1", 값 = "1", 시트명 = None):
        """
        :param 셀 : 저장될 셀위치 (예 : B3)
        :param 값 : 셀에 들어갈 값 (1개의 값 또는 tuple, list, pandas.Dataframe, numpy.ndarray)
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :return: 없음
        """
        excel, wb = self._open()    # 엑셀, 워크북 선언
        ws = self._시트명(wb, 시트명)  # 해당 시트 선택 없으면 시트생성, 시트명을 안 넣으면 첫번재 시트 선택

        # 데이터프레임일 경우 리스트로 바꿔줌
        if type(값) == pandas.core.frame.DataFrame:
            t1 = 값.reset_index()
            c = [list(t1.columns)]
            for i in range(0, t1.shape[0]):
                c.append(list(t1.iloc[i]))
            값 = c
        elif type(값) == tuple:
            값 = list(값)

        # 리스트 일경우 반복해서 넣어 줌
        if type(값) == list or type(값) == numpy.ndarray:
            c = 1
            r = 1
            for v in 값:
                if type(v) == list or type(v) == numpy.ndarray:
                    c = 1
                    for v1 in v:
                        ws.Range(셀).Offset(r, c).Value = v1
                        c = c + 1
                    r = r +1
                else:
                    ws.Range(셀).Offset(r, c).Value = v
                    c = c + 1
        else:
            ws.Range(셀).Value = 값

        self._close(ws, excel)

    def 셀값지우기(self, 셀="A1", 시트명 = None):
        """
        :param 셀 : 지울 셀위치 (예 : B3, A1:B3)
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :return: 없음
        """
        excel, wb = self._open()    # 엑셀, 워크북 선언
        ws = self._시트명(wb, 시트명)  # 해당 시트 선택 없으면 시트생성, 시트명을 안 넣으면 첫번재 시트 선택
        ws.Range(셀).ClearContents()
        self._close(ws, excel)



    def 이미지파일넣기(self,셀, 파일명, ColumnWidth=50, RowHeight = 150, 시트명 = None):
        r"""
        :param 셀 : 저장될 셀위치 (예 : B3)
        :param 파일명: 이미지 파일명 (예 : 'C:\Users\User\파이썬주피터\plot3.png' )
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :param ColumnWidth : 셀의 너비
        :param RowHeight : 셀의 높이
        :return: 없음
        """
        excel, wb = self._open()  # 엑셀, 워크북 선언
        ws = self._시트명(wb, 시트명)  # 해당 시트 선택 없으면 시트생성, 시트명을 안 넣으면 첫번재 시트 선택

        ws.Columns(ws.Range(셀).Column).ColumnWidth = ColumnWidth
        ws.Rows(ws.Range(셀).Row).RowHeight = RowHeight

        L = ws.Range(셀).Left
        T = ws.Range(셀).Top
        W = ws.Range(셀).Width
        H = ws.Range(셀).Height

        ws.Shapes.AddPicture(파일명, False, True, L, T, W,H).Placement = 1
        self._close(ws, excel)


    def 그래프넣기(self,셀, plt, ColumnWidth=50, RowHeight = 150, 시트명=None):
        """
        :param 셀: 저장될 셀위치 (예 : B3)
        :param plt: 그래프 object (예 :  matplotlib.pyplot )
        :param 시트명 : 입력하면 해당 시트에 입력 안하면 첫번째 시트에 값이 들어감
        :param ColumnWidth : 셀의 너비
        :param RowHeight : 셀의 높이
        :return: 없음
        """
        excel, wb = self._open()  # 엑셀, 워크북 선언
        ws = self._시트명(wb, 시트명)  # 해당 시트 선택 없으면 시트생성, 시트명을 안 넣으면 첫번재 시트 선택
        ws.Select()

        ws.Columns(ws.Range(셀).Column).ColumnWidth = ColumnWidth
        ws.Rows(ws.Range(셀).Row).RowHeight = RowHeight

        buf = io.BytesIO()
        plt.savefig(buf, format="svg")

        img = QImage.fromData(buf.getvalue())
        QApplication.clipboard().setImage(img)

        ws.Range(셀).Select()
        ws.Paste()

        cnt = ws.Shapes.Count
        ws.Shapes(cnt).LockAspectRatio = 0
        ws.Shapes(cnt).Height = ws.Range(셀).Height
        ws.Shapes(cnt).Width = ws.Range(셀).Width
        ws.Shapes(cnt).Placement = 1
        self._close(ws, excel)

