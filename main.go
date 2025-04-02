package main

import (
	"fmt"
	"regexp"
	"strings"
)

// 전역 변수로 컴파일된 정규식 패턴들을 저장
var (
	// 숫자만 포함하는지 검사하는 패턴
	digitOnlyPattern = regexp.MustCompile(`^\d+$`)

	// 카드번호 패턴 (13-19자리 숫자, 공백/점/대시 허용)
	cardNumberPattern = regexp.MustCompile(`\b\d{4}[\s\.-]?\d{4}[\s\.-]?\d{4}[\s\.-]?\d{4}\b`)
)

// Luhn 알고리즘을 사용하여 카드번호의 유효성을 검사합니다.
func isValidLuhn(number string) bool {
	// 공백, 점, 대시 제거
	number = strings.ReplaceAll(number, " ", "")
	number = strings.ReplaceAll(number, ".", "")
	number = strings.ReplaceAll(number, "-", "")

	// 숫자가 아닌 문자가 있으면 false 반환
	if !digitOnlyPattern.MatchString(number) {
		return false
	}

	// 디버깅을 위한 출력
	fmt.Printf("검증 중인 숫자: %s\n", number)

	var sum int
	nDigits := len(number)

	for i := nDigits - 1; i >= 0; i-- {
		d := int(number[i] - '0')
		if (nDigits-i)%2 == 0 {
			d = d * 2
			if d > 9 {
				d = d - 9
			}
		}
		sum += d
	}

	valid := sum%10 == 0
	fmt.Printf("합계: %d, 유효성: %v\n", sum, valid)
	return valid
}

func findCardNumbers(text string) []string {
	// 디버깅을 위한 출력
	fmt.Println("찾은 모든 매치:")
	matches := cardNumberPattern.FindAllString(text, -1)
	for _, match := range matches {
		fmt.Printf("매치: %s\n", match)
	}

	validCards := make([]string, 0)
	for _, match := range matches {
		if isValidLuhn(match) {
			validCards = append(validCards, match)
		}
	}

	return validCards
}

func main() {
	// 테스트용 텍스트
	text := `다음은 유효한 카드번호입니다: 4532 0151 2345 6789
그리고 이것도: 4532.0151.2345.6789
이것도: 4532-0151-2345-6789
이것은 유효하지 않은 번호입니다: 4532 0151 2345 6780
어쩌라고5320-9271-00708532 후후`

	cardNumbers := findCardNumbers(text)
	fmt.Println("발견된 유효한 카드번호:")
	for _, card := range cardNumbers {
		fmt.Printf("- %s\n", card)
	}
}
