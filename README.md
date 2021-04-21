# python

## python 3.8 버전 이용

*    KDHcalcClass.calc(upche, fmul, dateC, dateB, dateA, option: str = 'y')
*    """ 산식(&00-0001C&+...+&00-0005C&)을 금액으로 치환해서 계산 결과를 리턴해줌
*    ex) (&11-0001C&)+&12-0001C&
*        -> 100+&12-0001C&
*        -> 100+200
*        -> 300 리턴
*
*    산식예외처리
*    math library를 이용 
*        log,exp,log10,pow,sqrt 계산
*    별도 함수를 만들어 처리 
*        if, or, and
*    등호를 python 문법으로 치환
*        = -> ==
*
*    최종계산 eval 함수 이용
