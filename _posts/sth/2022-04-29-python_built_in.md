---
title:  "파이썬 내장함수"

categories:
  - Something_else
tags:
  - []

toc: true
toc_sticky: true

date: 2022-04-29
last_modified_at: 2022-04-29
---
# 답은 document에 있다..!  
<https://docs.python.org/ko/3/library/index.html>
# 파이썬 내장 함수
python 3.x 기준으로  
유용하게 사용했던 ~ 유용해보이는 내장함수, 기본모듈, 클래스 등을 기록하려 한다.  

위에서의 유용함의 의미는  
'메모리의 효율적이되, 코드가 간결해지는 사용할 수 있음' 혹은  
'정확한 표현을 간결하게 작성' 정도가 되겠다.  

## copy 모듈  
### copy.copy()  
얕은 복사 (주소를 대입하는 것) 를 한다.  

### copy.deepcopy()  
깊은 복사 (새롭게 주소를 생성) 를 한다.  

## enum  
열거형 데이터형을 만들어 쓸 수 있다.  

~~~
from enum import Enum
class Color(Enum):
  RED = 1,
  YELLOW = 2,
  BLUE = 3

member = Color.RED
print(member.name)
# 'RED'
print(member.value)
# 1
~~~  

## itertools 모듈   

### count()  
count(10)은 10, 11, 12 ....를,  
count(10,2)는 10, 12, 14 ... 를 무한으로 리턴함   

### cycle()  
cycle('asdf')는 a,s,d,f,a,s,d,f 를 무한으로 리턴함.   

### pairwise()  
pairwise('abcdefg')는 ab,bc,cd,de,ef,fg 를 리턴함.  

### product()  
product('abcd', 3)은 for i in 'abcd' 를 3중으로 붙인 값을 리턴함.  
aaa,aab,aac, ... ddd  

### permutations()  
순서를 고려한 모든 조합을 고른다.  
permutations('abc', 2) 는 ab, ac, ba, bc, ca, cb 를 리턴   

### combinations(), combinations_with_replacement()  
순서를 고려하지 않는 모든 조합을 만든다.  
combinations('abc', 2) 는 ab, ac, bc 를 리턴  
combinations_with_replacement 는 중복을 허용한다.  
위 예시와 같이 넣으면, aa, ab, ac, bb, bc, cc 를 리턴  




<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
