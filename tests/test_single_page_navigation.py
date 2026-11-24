"""Verify old page URLs redirect to the unified, working main interface."""
import http.client, pathlib, re, socket, subprocess, sys, time, unittest, urllib.parse
ROOT=pathlib.Path(__file__).resolve().parents[1]

class OnePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(("127.0.0.1",0))
            cls.port=s.getsockname()[1]
        cls.proc=subprocess.Popen([sys.executable,"server.py","--port",str(cls.port)],
            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(90):
            try:
                c=http.client.HTTPConnection("127.0.0.1",cls.port,timeout=1)
                c.request("GET","/")
                rsp=c.getresponse()
                if rsp.status==200:
                    rsp.read();c.close()
                    break
            except OSError:
                time.sleep(.06)
        else:
            raise AssertionError("Server failed to start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:cls.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:cls.proc.kill();cls.proc.wait()

    def request(self,path):
        c=http.client.HTTPConnection("127.0.0.1",self.port,timeout=4)
        c.request("GET",path)
        response=c.getresponse()
        data=response.read()
        result=(response.status,dict(response.getheaders()),data)
        c.close()
        return result

    def test_one_html_document_on_disk(self):
        self.assertEqual([x.name for x in ROOT.glob("*.html")],["index.html"])
    def test_old_studio_bookmark_preserves_scenario(self):
        status,headers,data=self.request("/research_studio.html?scenario_id=scenario_2")
        self.assertEqual(status,302)
        self.assertEqual(headers["Location"],"/?scenario_id=scenario_2#respondent")
        self.assertEqual(data,b"")
    def test_old_benchmark_bookmark_redirects(self):
        status,headers,data=self.request("/strict_benchmark.html")
        self.assertEqual(status,302)
        self.assertEqual(headers["Location"],"/#benchmark")
        self.assertEqual(data,b"")
    def test_root_and_index_share_content_and_current_version(self):
        status,headers,body=self.request("/")
        code,_,index=self.request("/index.html")
        self.assertEqual((status,code),(200,200))
        self.assertEqual(body,index)
        self.assertIn(b'content="1.9.3"',body)
        self.assertIn(b'Fast-Grounded',body)
        self.assertIn(b'SpokenTruth',body)
    def test_allowed_config_asset_and_private_source_protected(self):
        code,_,data=self.request("/assets/style.css")
        self.assertEqual(code,200)
        code,_,data=self.request("/server.py")
        self.assertEqual(code,403)
        self.assertTrue((ROOT/"resources/benchmark/benchmark.json").is_file())
        self.assertFalse((ROOT/"research_studio.html").exists())
        self.assertFalse((ROOT/"strict_benchmark.html").exists())

if __name__=="__main__": unittest.main()
