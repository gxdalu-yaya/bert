javac -cp .:./lib/* BertInference.java
java -cp .:./lib/* -Djava.library.path=./jni/ BertInference ./model/1541672157/
