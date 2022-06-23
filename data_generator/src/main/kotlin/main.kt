package com.github.pwoicik.psi

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.apache.commons.math3.analysis.integration.RombergIntegrator
import java.io.File
import kotlin.random.Random

fun main() = runBlocking {
    val integrator = RombergIntegrator()
    List(30_000) {
        async {
            val (a, b, c) = List(3) { Random.nextDouble(-1000.0, 1000.0) }
            val result = integrator
                .integrate(1_000, { x -> (a * x * x) + (b * x) + c }, 0.0, 1.0)
                .toBigDecimal()
                .toPlainString()
            "$a,$b,$c,$result\n"
        }
    }
        .awaitAll()
        .forEach(
            File("data.csv").apply {
                writeText("a,b,c,y\n")
            }::appendText
        )
}
