const calculateWordCount = (text: string): number => {
  if (!text) return 0;
  const cleanText = text
    .replace(/#{1,6}\s*/g, '')
    .replace(/[-*+]\s/g, '')
    .replace(/\d+\.\s/g, '')
    .replace(/```.*?\n/g, '')
    .replace(/```/g, '')
    .replace(/>/g, '')
    .replace(/\/\/\s*/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/[*_~]{1,2}([^*_~]+)[*_~]{1,2}/g, '$1 ')
    .replace(/#{1,6}/g, '')
    .replace(/[-*+]+/g, '')
    .replace(/---+/g, '')
    .replace(/[.,;:!?"'()]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const words = cleanText.split(/\s+/).filter(word => word.length > 0);
  return words.length;
};

const testCases = [
  { lang: 'English', text: 'The quick brown fox jumps over the lazy dog.' },
  { lang: 'Spanish', text: 'Me gusta escribir y aprender idiomas.' },
  { lang: 'German', text: 'Ich liebe Programmieren und Schreiben.' },
  { lang: 'Arabic', text: 'أنا أحب البرمجة والكتابة.' },
  { lang: 'Russian', text: 'Я люблю программировать и писать.' },
  { lang: 'Hindi', text: 'मुझे प्रोग्रामिंग और लेखन पसंद है।' },
  { lang: 'Chinese', text: '我喜欢学习和写作。' },
  { lang: 'Japanese', text: '私は日本語で書くのが好きです。' },
  { lang: 'Korean', text: '나는 한국어로 글을 씁니다.' },
  { lang: 'Thai', text: 'ฉันชอบเขียนและเรียนรู้สิ่งใหม่ ๆ' },
  { lang: 'Vietnamese', text: 'Tôi thích viết và học hỏi.' },
  { lang: 'Markdown', text: '# Header\n\n- Item one\n- Item two\n\nSome **bold** and _italic_ text.' }
];

for (const { lang, text } of testCases) {
  const count = calculateWordCount(text);
  console.log(`${lang} | Word Count: ${count}`);
}

