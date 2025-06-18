const calculateWordCount = (text: string): number => {
  if (!text) return 0;

  const nonSpaceCharPattern = /[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/g; // CJK: Chinese, Japanese, Korean
  const nonSpaceWords = text.match(nonSpaceCharPattern)?.length || 0;

  const cleanText = text
    .replace(/#{1,6}\s*/g, '') // Headers
    .replace(/[-*+]\s/g, '') // Unordered list
    .replace(/\d+\.\s/g, '') // Ordered list
    .replace(/```.*?\n/g, '') // Code fence open
    .replace(/```/g, '') // Code fence close
    .replace(/>/g, '') // Blockquote
    .replace(/\/\/\s*/g, ' ') // Comments
    .replace(/`([^`]+)`/g, '$1') // Inline code
    .replace(/[*_~]{1,2}([^*_~]+)[*_~]{1,2}/g, '$1 ') // Formatting
    .replace(/#{1,6}/g, '')
    .replace(/[-*+]+/g, '')
    .replace(/---+/g, '')
    .replace(/[.,;:!?"'()]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const spaceDelimitedWords = cleanText.split(/\s+/).filter(word => word.length > 0).length;

  return spaceDelimitedWords + nonSpaceWords;
};

const testCases = [
  { lang: 'English', text: 'The quick brown fox jumps over the lazy dog.' },
  { lang: 'Spanish', text: 'Me gusta escribir y aprender idiomas.' },
  { lang: 'German', text: 'Ich liebe das Schreiben und Lernen von Sprachen.' },
  { lang: 'Russian', text: 'Я люблю писать и изучать языки.' },
  { lang: 'Arabic', text: 'أنا أحب الكتابة وتعلم اللغات.' },
  { lang: 'Hindi', text: 'मुझे लिखना और भाषाएँ सीखना पसंद है।' },
  { lang: 'Chinese', text: '我喜欢学习和写作。' },
  { lang: 'Japanese', text: '私は日本語で書くのが好きです。' },
  { lang: 'Korean', text: '나는 한국어로 글을 쓰는 것을 좋아합니다.' },
  { lang: 'Markdown', text: '# Header\n\n- Item one\n- Item two\n\nSome **bold** and _italic_ text.' },
];

console.log('🧪 Multilingual Word Count Test:\n');

for (const { lang, text } of testCases) {
  const count = calculateWordCount(text);
  console.log(`${lang.padEnd(10)} | Word Count: ${count} | Text: "${text}"`);
}

