CC = mpicc
CFLAGS = -O2

PROJ = main
OBJS = main.o

run : $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean
