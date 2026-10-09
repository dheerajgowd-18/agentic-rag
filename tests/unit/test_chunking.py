import unittest
from app.ingestion.chunking.splitter import chunk_text


class TestChunking(unittest.TestCase):
    def test_empty_and_whitespace_input(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  \n  "), [])
        self.assertEqual(chunk_text(None), [])

    def test_text_shorter_than_chunk_size(self):
        short = "This is a short single-sentence text."
        chunks = chunk_text(short, chunk_size=200, chunk_overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short)

    def test_multiple_paragraphs(self):
        text = (
            "Paragraph one is about Kubernetes clusters and node management.\n\n"
            "Paragraph two focuses on Intel NIC hardware and SR-IOV drivers.\n\n"
            "Paragraph three covers software-defined networking and BGP routing."
        )
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 100)
            self.assertTrue(len(c.strip()) > 0)

    def test_long_paragraph_sentence_splitting(self):
        long_para = (
            "Kubernetes coordinates a highly available cluster of computers that are connected to work as a single unit. "
            "The abstractions in Kubernetes allow you to deploy containerized applications to a cluster without tying them specifically to individual machines. "
            "To make use of this new model of deployment, applications need to be packaged in a way that decouples them from individual hosts."
        )
        chunks = chunk_text(long_para, chunk_size=120, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 120)

    def test_text_without_normal_delimiters(self):
        # Continuous alphanumeric string without spaces or newlines
        continuous = "A" * 350
        chunks = chunk_text(continuous, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 100)
        # Verify no data loss in concatenation coverage
        total_len = sum(len(c) for c in chunks)
        self.assertGreaterEqual(total_len, 350)

    def test_very_long_individual_tokens(self):
        token_text = "start " + ("X" * 250) + " end"
        chunks = chunk_text(token_text, chunk_size=100, chunk_overlap=10)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_different_chunk_and_overlap_sizes(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks_1 = chunk_text(text, chunk_size=40, chunk_overlap=10)
        chunks_2 = chunk_text(text, chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks_1), len(chunks_2))

    def test_overlap_greater_than_chunk_size_safety(self):
        # When overlap >= chunk_size, function should adjust safely rather than crash or loop
        text = "Sample text for chunking safety check."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=100)
        self.assertTrue(len(chunks) > 0)

    def test_invalid_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("Hello", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("Hello", chunk_size=-10)


if __name__ == "__main__":
    unittest.main()
