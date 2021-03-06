### Next-Generation Security Intelligence Platform
This project aims to provide demonstrations of a security intelligence platform boosted by machine learning.

#### Installation
In order to run the demo, pull the repo and go to the root, then run the following command in your terminal to install dependencies.

```bash
pip install -r requirements.txt
```

Add project to your `PYTHONPATH`.

Afterwards, go to project root directory, and run

```bash
python run.py
```

If console gives you

```bash
[02/Aug/2017:13:28:53] ENGINE Bus STARTING
[02/Aug/2017:13:28:53] ENGINE Started monitor thread 'Autoreloader'.
[02/Aug/2017:13:28:53] ENGINE Started monitor thread '_TimeoutMonitor'.
[02/Aug/2017:13:28:53] ENGINE Serving on http://127.0.0.1:8080
[02/Aug/2017:13:28:53] ENGINE Bus STARTED
```

It means web server is sucessfully launchend, you can then browse to [http://127.0.0.1:8080](http://127.0.0.1:8080) for the demo.
