import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.agents.state import reduce_messages
from app.agents.graph import rag_agent


class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_reduce_messages_normal_accumulation(self):
        existing = [{"role": "user", "content": "hello"}]
        update = [{"role": "assistant", "content": "hi"}]
        res = reduce_messages(existing, update)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["content"], "hello")
        self.assertEqual(res[1]["content"], "hi")

    def test_reduce_messages_empty_list_does_not_wipe(self):
        # Critical regression fix: empty list update must NOT clear existing conversation
        existing = [{"role": "user", "content": "my name is Alice"}]
        res = reduce_messages(existing, [])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "my name is Alice")

    def test_reduce_messages_clear_marker(self):
        existing = [{"role": "user", "content": "my name is Alice"}]
        update = [{"role": "system", "content": "__CLEAR__"}]
        res = reduce_messages(existing, update)
        self.assertEqual(len(res), 0)

    def test_thread_state_isolation(self):
        config_a = {"configurable": {"thread_id": "thread_user_alice"}}
        config_b = {"configurable": {"thread_id": "thread_user_bob"}}

        rag_agent.update_state(config_a, {"messages": [{"role": "user", "content": "I am Alice"}]})
        rag_agent.update_state(config_b, {"messages": [{"role": "user", "content": "I am Bob"}]})

        state_a = rag_agent.get_state(config_a)
        state_b = rag_agent.get_state(config_b)

        msgs_a = state_a.values.get("messages", [])
        msgs_b = state_b.values.get("messages", [])

        self.assertEqual(msgs_a[-1]["content"], "I am Alice")
        self.assertEqual(msgs_b[-1]["content"], "I am Bob")

    def test_clear_memory_endpoint(self):
        config = {"configurable": {"thread_id": "thread_to_clear"}}
        rag_agent.update_state(config, {"messages": [{"role": "user", "content": "Forget me"}]})

        resp = self.client.post("/memory/clear", json={"thread_id": "thread_to_clear"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cleared")

        state = rag_agent.get_state(config)
        self.assertEqual(len(state.values.get("messages", [])), 0)


    def test_concurrent_different_threads_isolation(self):
        import concurrent.futures

        def worker(thread_idx: int):
            t_id = f"concurrent_thread_{thread_idx}"
            cfg = {"configurable": {"thread_id": t_id}}
            msg_content = f"Message from user {thread_idx}"
            rag_agent.update_state(cfg, {"messages": [{"role": "user", "content": msg_content}]})
            state = rag_agent.get_state(cfg)
            msgs = state.values.get("messages", [])
            return thread_idx, msgs

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(16)]
            for fut in concurrent.futures.as_completed(futures):
                idx, msgs = fut.result()
                self.assertGreaterEqual(len(msgs), 1)
                self.assertEqual(msgs[-1]["content"], f"Message from user {idx}")

    def test_concurrent_same_thread_updates(self):
        import concurrent.futures
        shared_thread_id = "shared_concurrent_thread"
        cfg = {"configurable": {"thread_id": shared_thread_id}}

        # Clear first
        rag_agent.update_state(cfg, {"messages": [{"role": "system", "content": "__CLEAR__"}]})

        def append_msg(i: int):
            rag_agent.update_state(cfg, {"messages": [{"role": "user", "content": f"turn_{i}"}]})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(append_msg, range(10)))

        state = rag_agent.get_state(cfg)
        msgs = state.values.get("messages", [])
        self.assertEqual(len(msgs), 10)

    def test_memory_clearing_during_concurrent_use(self):
        import concurrent.futures
        thread_to_wipe = "thread_wiped"
        thread_to_keep = "thread_kept"

        cfg_wipe = {"configurable": {"thread_id": thread_to_wipe}}
        cfg_keep = {"configurable": {"thread_id": thread_to_keep}}

        rag_agent.update_state(cfg_wipe, {"messages": [{"role": "user", "content": "Wipe me"}]})
        rag_agent.update_state(cfg_keep, {"messages": [{"role": "user", "content": "Keep me"}]})

        def run_wipe():
            resp = self.client.post("/memory/clear", json={"thread_id": thread_to_wipe})
            return resp.status_code

        def run_keep_query():
            state = rag_agent.get_state(cfg_keep)
            return len(state.values.get("messages", []))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(run_wipe)
            f2 = executor.submit(run_keep_query)

            self.assertEqual(f1.result(), 200)
            self.assertEqual(f2.result(), 1)

        # Confirm wiped
        state_wiped = rag_agent.get_state(cfg_wipe)
        self.assertEqual(len(state_wiped.values.get("messages", [])), 0)


if __name__ == "__main__":
    unittest.main()
