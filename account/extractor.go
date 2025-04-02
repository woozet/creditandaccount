package main

import (
	"fmt"
	"regexp"
	"strings"
)

// AccountExtractor는 계좌번호 추출을 위한 구조체입니다.
type AccountExtractor struct {
	// 계좌번호 패턴 (10-16자리 숫자, 공백/대시/온점 허용)
	pattern *regexp.Regexp
}

// 전역 변수로 컴파일된 정규식 패턴을 저장
var accountPattern = regexp.MustCompile(`\b\d[\d\s\.-]{8,14}\d\b`)

// 숫자만 추출하는 함수를 전역 변수로 정의
var digitOnly = func(r rune) rune {
	if r >= '0' && r <= '9' {
		return r
	}
	return -1
}

// NewAccountExtractor는 새로운 AccountExtractor를 생성합니다.
func NewAccountExtractor() *AccountExtractor {
	return &AccountExtractor{
		pattern: accountPattern,
	}
}

// ExtractAccounts는 텍스트에서 계좌번호를 추출합니다.
func (e *AccountExtractor) ExtractAccounts(text string) []string {
	// 패턴 매치 찾기
	matches := e.pattern.FindAllString(text, -1)

	// 결과 저장을 위한 슬라이스
	accounts := make([]string, 0, len(matches))

	// 각 매치에 대해 처리
	for _, match := range matches {
		// 특수문자 제거 (재사용된 함수 사용)
		digits := strings.Map(digitOnly, match)

		// 결과 추가
		accounts = append(accounts, digits)
	}

	return accounts
}

func main() {
	// 테스트용 텍스트
	text := `다음은 계좌번호입니다: 110-123-456789
	그리고 이것도: 110.123.456790
	이것도: 110 123 456791
	이것은 계좌번호가 아닙니다: 123-45-6789
	이것도 계좌번호입니다: 123-45-6789012
	이것도: 123.45.6789013
	이것도: 123 45 6789014
	이것도: 123-456-7890123
	이것도: 3333-12-3456789`

	// 추출기 생성
	extractor := NewAccountExtractor()

	// 계좌번호 추출
	accounts := extractor.ExtractAccounts(text)

	// 결과 출력
	fmt.Println("발견된 계좌번호:")
	for i, account := range accounts {
		fmt.Printf("%d. %s\n", i+1, account)
	}
}
